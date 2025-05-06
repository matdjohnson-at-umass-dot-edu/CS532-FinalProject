echo "entrypoint script started, sleeping for 60s"
sleep 60;
echo "executing mysql"
mysql --protocol=TCP --host=pyspark-notebook-mysql --port=3306 --user=root --password=root < /files/init.sql
echo "mysql execution completed"
