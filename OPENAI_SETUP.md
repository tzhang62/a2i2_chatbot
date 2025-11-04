# OpenAI API Setup Guide

This document provides quick instructions for setting up and using the OpenAI API with this project.

## Quick Start

1. **Get an OpenAI API Key:**
   - Visit https://platform.openai.com/
   - Sign up or log in to your account
   - Navigate to "API keys" in the left sidebar
   - Click "Create new secret key"
   - Copy the key (you won't be able to see it again!)

2. **Create a `.env` file:**
   ```bash
   # In the project root directory (A2I2/)
   touch .env
   ```

3. **Add your API key to `.env`:**
   ```
   OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxx
   ```
   Replace `sk-proj-xxxxxxxxxxxxxxxxxxxxx` with your actual API key.

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the server:**
   ```bash
   cd backend
   uvicorn server:app --host 0.0.0.0 --port 8001
   ```

## Configuration Options

### Change the AI Model

By default, the system uses `gpt-4o-mini`. To use a different model:

1. Open `backend/ollama_0220_openai.py`
2. Find the `send_to_openai()` function (around line 293)
3. Change the `model` parameter:

```python
def send_to_openai(prompt: str, model: str = "gpt-4o-mini") -> str:
    """Query the OpenAI API with the given prompt."""
    try:
        response = client.chat.completions.create(
            model=model,  # Change this to "gpt-4o", "gpt-3.5-turbo", etc.
            messages=[{
                'role': 'user',
                'content': prompt,
            }],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error calling OpenAI API: {str(e)}")
        raise
```

### Available Models

- `gpt-4o-mini` (default) - Fast and cost-effective
- `gpt-4o` - Most capable, higher cost
- `gpt-4-turbo` - Balanced performance
- `gpt-3.5-turbo` - Fastest and cheapest

See https://platform.openai.com/docs/models for the full list and pricing.

### Adjust Generation Parameters

In the same `send_to_openai()` function, you can adjust:

- **temperature** (0.0-2.0): Controls randomness. Lower = more deterministic, Higher = more creative
- **max_tokens**: Maximum length of response

```python
response = client.chat.completions.create(
    model=model,
    messages=[{'role': 'user', 'content': prompt}],
    temperature=0.7,  # Adjust this (0.0-2.0)
    max_tokens=500    # Adjust this
)
```

## Cost Management

### Monitor Your Usage

- Check usage at: https://platform.openai.com/usage
- Set up billing alerts in your OpenAI account

### Estimated Costs (as of 2025)

For `gpt-4o-mini`:
- Input: $0.15 per 1M tokens
- Output: $0.60 per 1M tokens

A typical conversation (10-15 exchanges) uses approximately:
- Input: ~2,000 tokens ($0.0003)
- Output: ~500 tokens ($0.0003)
- **Total per conversation: ~$0.0006**

### Cost Saving Tips

1. Use `gpt-4o-mini` instead of `gpt-4o` (10x cheaper)
2. Reduce `max_tokens` if responses are too long
3. Implement caching for repeated queries (if applicable)

## Troubleshooting

### "OpenAI API key not found" Error

**Solution:** Make sure the `.env` file is in the project root directory and contains:
```
OPENAI_API_KEY=your_actual_key_here
```

### "Import openai could not be resolved" Warning

**Solution:** Install the OpenAI package:
```bash
pip install openai>=1.0.0
```

### "Rate limit exceeded" Error

**Solution:** 
- You've hit your API usage limit
- Check your plan at https://platform.openai.com/account/billing
- Upgrade your plan or wait for the rate limit to reset

### "Invalid API key" Error

**Solution:**
- Verify your API key is correct
- Make sure there are no extra spaces or quotes around the key in `.env`
- Generate a new API key if the old one was revoked

## Security Best Practices

1. **Never commit `.env` file:**
   - Add `.env` to your `.gitignore`
   - The file is already gitignored in this project

2. **Never share your API key:**
   - Don't post it in public forums or repositories
   - Don't include it in screenshots

3. **Rotate keys regularly:**
   - Generate new keys periodically
   - Delete old keys from OpenAI dashboard

4. **Use environment-specific keys:**
   - Different keys for development and production
   - Set usage limits on development keys

## Switching Back to Ollama

If you need to use the local Ollama model instead:

1. Edit `backend/server.py`:
   ```python
   # Change this line:
   from ollama_0220_openai import ...
   
   # To this:
   from ollama_0220 import ...
   ```

2. Install and start Ollama:
   ```bash
   # Install Ollama from https://ollama.ai
   ollama pull llama3.2:latest
   ollama serve
   ```

3. No API key needed for local models

## Support

For OpenAI API issues:
- Documentation: https://platform.openai.com/docs
- Support: https://help.openai.com/

For project-specific issues:
- Check the main README.md
- Review the code in `backend/ollama_0220_openai.py`

