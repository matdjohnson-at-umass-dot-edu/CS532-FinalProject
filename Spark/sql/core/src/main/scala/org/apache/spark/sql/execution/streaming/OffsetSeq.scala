/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.sql.execution.streaming

import org.json4s.NoTypeHints
import org.json4s.jackson.Serialization

import org.apache.spark.internal.Logging
import org.apache.spark.sql.RuntimeConfig
import org.apache.spark.sql.connector.read.streaming.{Offset => OffsetV2, SparkDataStream}
import org.apache.spark.sql.execution.streaming.state.{FlatMapGroupsWithStateExecHelper, StreamingAggregationStateManager, SymmetricHashJoinStateManager}
import org.apache.spark.sql.internal.SQLConf.{FLATMAPGROUPSWITHSTATE_STATE_FORMAT_VERSION, _}


/**
 * An ordered collection of offsets, used to track the progress of processing data from one or more
 * [[Source]]s that are present in a streaming query. This is similar to simplified, single-instance
 * vector clock that must progress linearly forward.
 */
case class OffsetSeq(offsets: Seq[Option[OffsetV2]], metadata: Option[OffsetSeqMetadata] = None) {

  /**
   * Unpacks an offset into [[StreamProgress]] by associating each offset with the ordered list of
   * sources.
   *
   * This method is typically used to associate a serialized offset with actual sources (which
   * cannot be serialized).
   */
  def toStreamProgress(sources: Seq[SparkDataStream]): StreamProgress = {
    assert(sources.size == offsets.size, s"There are [${offsets.size}] sources in the " +
      s"checkpoint offsets and now there are [${sources.size}] sources requested by the query. " +
      s"Cannot continue.")
    new StreamProgress ++ sources.zip(offsets).collect { case (s, Some(o)) => (s, o) }
  }

  override def toString: String =
    offsets.map(_.map(_.json).getOrElse("-")).mkString("[", ", ", "]")
}

object OffsetSeq {

  /**
   * Returns a [[OffsetSeq]] with a variable sequence of offsets.
   * `nulls` in the sequence are converted to `None`s.
   */
  def fill(offsets: OffsetV2*): OffsetSeq = OffsetSeq.fill(None, offsets: _*)

  /**
   * Returns a [[OffsetSeq]] with metadata and a variable sequence of offsets.
   * `nulls` in the sequence are converted to `None`s.
   */
  def fill(metadata: Option[String], offsets: OffsetV2*): OffsetSeq = {
    OffsetSeq(offsets.map(Option(_)), metadata.map(OffsetSeqMetadata.apply))
  }
}


/**
 * Contains metadata associated with a [[OffsetSeq]]. This information is
 * persisted to the offset log in the checkpoint location via the [[OffsetSeq]] metadata field.
 *
 * @param batchWatermarkMs: The current eventTime watermark, used to
 * bound the lateness of data that will processed. Time unit: milliseconds
 * @param batchTimestampMs: The current batch processing timestamp.
 * Time unit: milliseconds
 * @param conf: Additional conf_s to be persisted across batches, e.g. number of shuffle partitions.
 */
case class OffsetSeqMetadata(
    batchWatermarkMs: Long = 0,
    batchTimestampMs: Long = 0,
    conf: Map[String, String] = Map.empty) {
  def json: String = Serialization.write(this)(OffsetSeqMetadata.format)
}

object OffsetSeqMetadata extends Logging {
  private implicit val format = Serialization.formats(NoTypeHints)
  /**
   * These configs are related to streaming query execution and should not be changed across
   * batches of a streaming query. The values of these configs are persisted into the offset
   * log in the checkpoint position.
   */
  private val relevantSQLConfs = Seq(
    SHUFFLE_PARTITIONS, STATE_STORE_PROVIDER_CLASS, STREAMING_MULTIPLE_WATERMARK_POLICY,
    FLATMAPGROUPSWITHSTATE_STATE_FORMAT_VERSION, STREAMING_AGGREGATION_STATE_FORMAT_VERSION,
    STREAMING_JOIN_STATE_FORMAT_VERSION, STATE_STORE_COMPRESSION_CODEC,
    STATE_STORE_ROCKSDB_FORMAT_VERSION, STATEFUL_OPERATOR_USE_STRICT_DISTRIBUTION,
    PRUNE_FILTERS_CAN_PRUNE_STREAMING_SUBPLAN
  )

  /**
   * Default values of relevant configurations that are used for backward compatibility.
   * As new configurations are added to the metadata, existing checkpoints may not have those
   * confs. The values in this list ensures that the confs without recovered values are
   * set to a default value that ensure the same behavior of the streaming query as it was before
   * the restart.
   *
   * Note, that this is optional; set values here if you *have* to override existing session conf
   * with a specific default value for ensuring same behavior of the query as before.
   */
  private val relevantSQLConfDefaultValues = Map[String, String](
    STREAMING_MULTIPLE_WATERMARK_POLICY.key -> MultipleWatermarkPolicy.DEFAULT_POLICY_NAME,
    FLATMAPGROUPSWITHSTATE_STATE_FORMAT_VERSION.key ->
      FlatMapGroupsWithStateExecHelper.legacyVersion.toString,
    STREAMING_AGGREGATION_STATE_FORMAT_VERSION.key ->
      StreamingAggregationStateManager.legacyVersion.toString,
    STREAMING_JOIN_STATE_FORMAT_VERSION.key ->
      SymmetricHashJoinStateManager.legacyVersion.toString,
    STATE_STORE_COMPRESSION_CODEC.key -> "lz4",
    STATEFUL_OPERATOR_USE_STRICT_DISTRIBUTION.key -> "false",
    PRUNE_FILTERS_CAN_PRUNE_STREAMING_SUBPLAN.key -> "true"
  )

  def apply(json: String): OffsetSeqMetadata = Serialization.read[OffsetSeqMetadata](json)

  def apply(
      batchWatermarkMs: Long,
      batchTimestampMs: Long,
      sessionConf: RuntimeConfig): OffsetSeqMetadata = {
    val confs = relevantSQLConfs.map { conf => conf.key -> sessionConf.get(conf.key) }.toMap
    OffsetSeqMetadata(batchWatermarkMs, batchTimestampMs, confs)
  }

  /** Set the SparkSession configuration with the values in the metadata */
  def setSessionConf(metadata: OffsetSeqMetadata, sessionConf: RuntimeConfig): Unit = {
    OffsetSeqMetadata.relevantSQLConfs.map(_.key).foreach { confKey =>

      metadata.conf.get(confKey) match {

        case Some(valueInMetadata) =>
          // Config value exists in the metadata, update the session config with this value
          val optionalValueInSession = sessionConf.getOption(confKey)
          if (optionalValueInSession.isDefined && optionalValueInSession.get != valueInMetadata) {
            logWarning(s"Updating the value of conf '$confKey' in current session from " +
              s"'${optionalValueInSession.get}' to '$valueInMetadata'.")
          }
          sessionConf.set(confKey, valueInMetadata)

        case None =>
          // For backward compatibility, if a config was not recorded in the offset log,
          // then either inject a default value (if specified in `relevantSQLConfDefaultValues`) or
          // let the existing conf value in SparkSession prevail.
          relevantSQLConfDefaultValues.get(confKey) match {

            case Some(defaultValue) =>
              sessionConf.set(confKey, defaultValue)
              logWarning(s"Conf '$confKey' was not found in the offset log, " +
                s"using default value '$defaultValue'")

            case None =>
              val valueStr = sessionConf.getOption(confKey).map { v =>
                s" Using existing session conf value '$v'."
              }.getOrElse { " No value set in session conf." }
              logWarning(s"Conf '$confKey' was not found in the offset log. $valueStr")

          }
      }
    }

    // SPARK-51187: This incorrect config is not added in the relevantSQLConfs, but the
    // metadata in the offset log may have this if the batch ran from Spark 3.5.4.
    // We need to pick the value from the metadata and set it in the new config.
    // This also leads the further batches to have a correct config in the offset log.
    metadata.conf.get("spark.databricks.sql.optimizer.pruneFiltersCanPruneStreamingSubplan") match {
      case Some(value) =>
        sessionConf.set(PRUNE_FILTERS_CAN_PRUNE_STREAMING_SUBPLAN.key, value)

      case _ =>
    }
  }
}
