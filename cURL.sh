# Example cURL for Copernicus Data Space Ecosystem OAuth
curl -X POST "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token" ^
  -H "Content-Type: application/x-www-form-urlencoded" ^
  -d "grant_type=client_credentials&client_id=sh-c8843217-c11c-4477-badd-13295ad2951b&client_secret=BoJiCTPPmGXJV7YUDmjBVXdQSUPn29PJ"

curl --request POST --url https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token --header 'content-type: application/x-www-form-urlencoded' --data 'grant_type=client_credentials&client_id=sh-9cce9a22-47ff-4124-81b7-9d8d48edf562' --data-urlencode 'client_secret=kneeUZWVVfh1z9MUmO9PPBR87ueFxPIO'

# Old Sentinel Hub endpoint (deprecated, do not use)
# curl --request POST --url https://services.sentinel-hub.com/oauth/token --header 'content-type: application/x-www-form-urlencoded' --data 'grant_type=client_credentials&client_id=sh-xxxx' --data-urlencode 'client_secret=xxxx'