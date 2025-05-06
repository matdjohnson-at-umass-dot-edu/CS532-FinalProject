# CS532 Final Project

This is a repository containing the source code for a final project submission for CS 532 - Systems for Data Science.

The repository contains the code used to run a local Spark instance and execute a PySpark job on the instance.

The PySpark job builds an inverted index from the WIKIR dataset, which is then evaluated using the BM25 retrieval algorithm and the query relevance data from the WIKIR database.

The Spark instance code is in the ./Spark directory. This is a clone of the Spark sources, and is used in order to ensure version consistency. Note that the repository is frozen given that the git directory has been renamed to `.gitoriginal`.

The PySpark job code is in the ./PySpark directory. The job is implemented in the `CS532-FinalProjectProgram.py` file.

## Setup and usage

1) Install Java 17. This can be done using apt-get on Ubuntu or brew on OSX. Ensure that the shell profile configuration file (`~/.bash_profile`) contains the environment variable JAVA_HOME and that it is set to the Java 17 install directory. Any shell used to install Java 17 should be exited, and a new shell opened 
2) Create a Python virtual environment, activate the environment, and install the job dependencies. An example command execution is:
    ```
    python3 -mvenv ./PySpark;
    source ./PySpark/bin/activate;
    pip3 install -r ./PySpark/requirements.txt;
    ```
3) Build Spark and install Spark to the Python environment.
    ```
    ./Spark/dev/make-distribution.sh --name custom-spark --pip --tgz -Phive -Phive-thriftserver -Pjvm-profiler -Pyarn -Pkubernetes
    pip3 install ./Spark/dist/python/
    ```
4) Start the Spark cluster
   ```
   ./Spark/sbin/start-master.sh
   ./Spark/sbin/start-worker.sh <open localhost:8080 and copy the URI at the top of the screen>
   ```
5) Start the Docker dependencies (A MySQL database for storing the inverted index when built):
   ```
   docker compose -f docker/docker-compose.yml up --build -d
   ```
6) Run the Python Program
   ```
   # copy the URI from the Spark UI and paste it in to the spark_host variable in the python file
   # configure the program max_cores and max_mem_mb variables consistent with what is displayed in the Spark UI (or a lesser amount in integers)
   # the program runs on a 300k subsample of the main WIKIR dataset
   python3 ./PySpark/CS532-FinalProjectProgram.py
   ```

When done running the program, ensure that the JAVA_HOME variable is reverted, that the Spark cluster is stopped, and that the docker containers have been stopped deleted.

```
./Spark/sbin/stop-worker.sh
./Spark/sbin/stop-master.sh
docker compose -f docker/docker-compose.yml down
```

There is also a Jupyter notebook containing the same code as the python program, which can be executed on Colab. The notebook is at `./Pyspark/CS532-FinalProjectNotebook.ipynb`. The notebook is also available on Google Drive (ref: [Google Drive link](https://drive.google.com/drive/folders/14Y9p6RUPPtwTbFi_WvHVOiVapHydLt9f?usp=sharing)).

Execution of the Jupyter notebook on Colab requires the Google Drive folder to be added as a shortcut under "My Drive". This can be done by selecting the folder title at the top of the Google Drive screen, opening the "Organization" submenu, selecting "Add Shortcut", then selecting "My Drive" from "All Places".

Colab execution is configured to be performed with the v2-8 TPU instance type, which has a total of 96 cores, and is available free at the time of this writing.

