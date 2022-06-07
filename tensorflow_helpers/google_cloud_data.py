from google.cloud import storage
# create storage client object
storage_client = storage.Client.from_service_account_json('JSON path which is generated from service account')
# get bucket object name
bucket = storage_client.get_bucket('bucket_name')
blob = bucket.get_blob('actual object name')
# get the actual data to be downloaded....
json_data = blob.download_as_string()
for blob in storage_client.list_blobs('bucket_name', prefix='folder name and add nested folder name'):
    wiki_paths.append(blob.name)
