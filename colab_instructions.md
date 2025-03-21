# Setting up Emergency Response Chatbot on Google Colab

This guide will help you set up the Emergency Response Chatbot backend on Google Colab.

## Prerequisites

1. A Google account
2. Access to Google Drive
3. The following files from your project:
   - `data_for_train/persona.json`
   - `data_for_train/dialogue_1.json`
   - `ollama_0220.py`

## Setup Instructions

1. **Prepare Your Files**
   - Create a folder in your Google Drive named `A2I2`
   - Copy your data files into this folder:
     - `persona.json` → `A2I2/data_for_train/persona.json`
     - `dialogue_1.json` → `A2I2/data_for_train/dialogue_1.json`

2. **Open Google Colab**
   - Go to [Google Colab](https://colab.research.google.com)
   - Create a new notebook
   - Rename it to "Emergency Response Chatbot Backend"

3. **Copy the Setup Code**
   - Copy the contents of `colab_setup.ipynb` into your Colab notebook
   - Make sure to run each cell in sequence

4. **Enable GPU**
   - Click on "Runtime" in the menu
   - Select "Change runtime type"
   - Under "Hardware accelerator", select "GPU"
   - Click "Save"

5. **Run the Notebook**
   - Run each cell in sequence by clicking the play button or pressing Shift+Enter
   - When prompted, authorize Google Drive access
   - The server will start on port 8000

6. **Get Your Backend URL**
   - After running the last cell, you'll see a URL like `http://0.0.0.0:8000`
   - To get your public URL, run this code in a new cell:
     ```python
     from google.colab import output
     output.register_callback('notebook.getUrl', lambda: print('Your backend URL:', 'https://your-colab-url'))
     ```

7. **Update Frontend Configuration**
   - Open your `docs/js/chat.js` file
   - Update the `API_BASE_URL` to your Colab backend URL
   - Commit and push the changes to GitHub

## Important Notes

- The Colab runtime will disconnect after 12 hours of inactivity
- You'll need to reconnect and run the setup again after disconnection
- Keep the Colab tab active to maintain the connection
- The GPU instance will be faster than the CPU-only version on Render

## Troubleshooting

1. **If the server fails to start:**
   - Check if port 8000 is already in use
   - Try using a different port (e.g., 8080)
   - Make sure all required packages are installed

2. **If you can't access the backend:**
   - Verify that the Colab runtime is active
   - Check if your frontend URL is correctly configured
   - Ensure CORS is properly configured in the backend

3. **If the model is slow:**
   - Make sure GPU is enabled
   - Check GPU memory usage
   - Consider using a smaller model variant

## Next Steps

1. Test the connection between your frontend and the Colab backend
2. Monitor the performance and adjust as needed
3. Consider setting up automatic reconnection scripts if needed

For any issues or questions, please refer to the project documentation or create an issue in the repository. 