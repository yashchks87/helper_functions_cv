# Author: Yash Choksi
# Date: Feb 17th 2022
# This codebook specifically mentions how to access data on colab from
# google cloud buckets and specifically data stored as private.

# Use this import to use this utility
from google.colab import auth

# This will generate one link in code cell and by clicking that link
# go to list of google accounts and select the account under which bucket 
# is getting hosted. After that one string will be there just copy it
# to the cell where one field is shown. Doing that it will let you use all 
# private data.
auth.authenticate_user()