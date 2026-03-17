#!/usr/bin/env python3
"""
OAuth Debugging Script for Sentinel Hub
Tests OAuth credentials with detailed error reporting
"""

import json
import os
import time
import requests
from datetime import datetime

def load_config():
    """Load credentials from config.json"""
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        return config['sentinel_hub']
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return None

def test_oauth_credentials(client_id, client_secret, max_retries=3, delay=60):
    """
    Test OAuth credentials with retry logic
    """
    url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    headers = {
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret
    }
    
    print(f"🔍 Testing OAuth credentials...")
    print(f"   Client ID: {client_id[:10]}...")
    print(f"   URL: {url}")
    print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    for attempt in range(max_retries):
        if attempt > 0:
            print(f"\n⏳ Retry attempt {attempt + 1}/{max_retries}")
            print(f"   Waiting {delay} seconds for propagation...")
            time.sleep(delay)
        
        try:
            print(f"\n📡 Attempt {attempt + 1}: Sending OAuth request...")
            response = requests.post(url, headers=headers, data=data, timeout=30)
            
            print(f"   Status Code: {response.status_code}")
            print(f"   Response Headers: {dict(response.headers)}")
            
            if response.status_code == 200:
                token_data = response.json()
                print(f"✅ SUCCESS! OAuth token received")
                print(f"   Token Type: {token_data.get('token_type')}")
                print(f"   Expires In: {token_data.get('expires_in')} seconds")
                print(f"   Access Token: {token_data.get('access_token', '')[:20]}...")
                return True
            else:
                print(f"❌ FAILED: HTTP {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"   Error: {error_data.get('error')}")
                    print(f"   Description: {error_data.get('error_description')}")
                    
                    # Specific error handling
                    if error_data.get('error') == 'invalid_client':
                        print(f"   💡 This usually means:")
                        print(f"      - Credentials are incorrect")
                        print(f"      - Client is not enabled")
                        print(f"      - Client is too new (wait 5-15 minutes)")
                        print(f"      - Account has restrictions")
                    
                except json.JSONDecodeError:
                    print(f"   Raw Response: {response.text}")
                
        except requests.exceptions.Timeout:
            print(f"❌ TIMEOUT: Request took too long")
        except requests.exceptions.ConnectionError:
            print(f"❌ CONNECTION ERROR: Cannot reach Copernicus Data Space OAuth endpoint")
        except Exception as e:
            print(f"❌ UNEXPECTED ERROR: {e}")
    
    return False

def check_sentinel_hub_status():
    """Check if Copernicus Data Space services are accessible"""
    print(f"\n🌐 Checking Copernicus Data Space service status...")
    
    try:
        # Test basic connectivity
        response = requests.get("https://sh.dataspace.copernicus.eu/", timeout=10)
        print(f"   Main service: ✅ Accessible (HTTP {response.status_code})")
    except Exception as e:
        print(f"   Main service: ❌ Not accessible ({e})")
    
    try:
        # Test OAuth endpoint specifically
        response = requests.get("https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token", timeout=10)
        print(f"   OAuth endpoint: ✅ Accessible (HTTP {response.status_code})")
    except Exception as e:
        print(f"   OAuth endpoint: ❌ Not accessible ({e})")

def main():
    print("🔐 Sentinel Hub OAuth Debug Tool")
    print("=" * 50)
    
    # Check service status first
    check_sentinel_hub_status()
    
    # Load credentials
    config = load_config()
    if not config:
        print("❌ Cannot load credentials. Exiting.")
        return
    
    client_id = config.get('client_id')
    client_secret = config.get('client_secret')
    
    if not client_id or not client_secret:
        print("❌ Missing client_id or client_secret in config")
        return
    
    # Test credentials with retry logic
    success = test_oauth_credentials(client_id, client_secret)
    
    print("\n" + "=" * 50)
    if success:
        print("✅ OAuth credentials are working!")
        print("   Your Python code should work now.")
    else:
        print("❌ OAuth credentials are not working.")
        print("\n💡 Troubleshooting steps:")
        print("   1. Wait 5-15 minutes for new client propagation")
        print("   2. Check if client is enabled in Sentinel Hub dashboard")
        print("   3. Verify your account is active")
        print("   4. Try regenerating the client secret")
        print("   5. Contact Sentinel Hub support if issues persist")

if __name__ == "__main__":
    main() 