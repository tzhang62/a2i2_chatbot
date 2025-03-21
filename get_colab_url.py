from google.colab import output
import requests
import time

def get_colab_url():
    """Get the public URL of the Colab notebook."""
    try:
        # Get the notebook URL
        notebook_url = output.eval_js('google.colab.kernel.proxyPort(8000)')
        print(f"Your backend is accessible at: {notebook_url}")
        return notebook_url
    except Exception as e:
        print(f"Error getting URL: {str(e)}")
        return None

def check_server_status(url):
    """Check if the server is running and accessible."""
    try:
        response = requests.get(f"{url}/")
        if response.status_code == 200:
            print("Server is running and accessible!")
            return True
        else:
            print(f"Server returned status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error checking server status: {str(e)}")
        return False

if __name__ == "__main__":
    print("Getting Colab URL...")
    url = get_colab_url()
    
    if url:
        print("\nChecking server status...")
        max_attempts = 5
        for attempt in range(max_attempts):
            if check_server_status(url):
                break
            print(f"Attempt {attempt + 1}/{max_attempts} failed. Retrying in 5 seconds...")
            time.sleep(5) 