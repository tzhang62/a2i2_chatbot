# New Characters Added - Documentation

## Summary

Successfully added 5 new characters to the Emergency Response Simulation system:
1. **Mary** - 67-year-old woman with dog
2. **Ben** - 29-year-old work-from-home tech
3. **Ana** - 42-year-old caregiver at senior center
4. **Tom** - 54-year-old high school teacher
5. **Mia** - 17-year-old high school robotics student

## What Was Updated

### Frontend Changes

#### 1. `/frontend/index.html`
- Added 5 new character cards to the person selection grid
- Now displays all 10 characters: Bob, Niki, Lindsay, Ross, Michelle, Mary, Ben, Ana, Tom, Mia
- Each card includes character name, description, and selection button

### Backend Changes

#### 2. `/backend/server.py`
- Added dialogue data loading for new characters:
  - `mary_data`
  - `ben_data`
  - `ana_data`
  - `tom_data`
  - `mia_data`
- Characters are now loaded at server startup

#### 3. `/data_for_train/persona.json`
- Already contained persona data for all new characters
- No changes needed (data was already present)

#### 4. `/data_for_train/character_lines.jsonl`
- Added dialogue lines for all 5 new characters
- Each character has:
  - `greetings` - Initial responses
  - `response_to_operator_greetings` - How they respond to operator
  - `progression` - Responses showing willingness to evacuate
  - `observations` - What they see/report about the fire
  - `general` - General conversational responses
  - `closing` - Goodbye messages

#### 5. `/data_for_train/characterlines.jsonl` (no underscore)
- Also updated with new character dialogue data
- Ensures compatibility with different backend modules

## Character Profiles

### Mary
- **Age:** 67
- **Key Traits:** Lives alone with dog Poppy, has arthritis, needs assistance
- **Evacuation Needs:** Van pickup required, cannot drive herself
- **Special Considerations:** Moves slowly, needs space for her dog

### Ben
- **Age:** 29
- **Key Traits:** Works from home, tech-savvy, has pet gecko
- **Evacuation Needs:** Can use e-bike or car, may need route guidance
- **Special Considerations:** May delay to save work, wants to bring pet

### Ana
- **Age:** 42
- **Key Traits:** Caregiver at senior center, responsible for multiple residents
- **Evacuation Needs:** Group transport van required for 8+ seniors
- **Special Considerations:** Won't leave until all residents are safe

### Tom
- **Age:** 54
- **Key Traits:** High school teacher, knows many people in town
- **Evacuation Needs:** Has own truck, but may delay to help others
- **Special Considerations:** Wants to volunteer/help neighbors

### Mia
- **Age:** 17
- **Key Traits:** Smart student, focused on robotics work
- **Evacuation Needs:** Can drive herself, may need alerts/guidance
- **Special Considerations:** May not notice alerts while working

## How the Characters Work

### Conversation Flow
All new characters follow a cooperative conversation pattern similar to Ross:

1. **Initial Contact** - They answer the call
2. **Situation Assessment** - They explain their situation/needs
3. **Assistance Request** - They ask for specific help
4. **Evacuation Agreement** - They agree to evacuate with proper support
5. **Closing** - They thank the operator and prepare to leave

### Behavior Differences from Original 5

- **More Cooperative:** Unlike Bob and Michelle who resist, new characters are willing to evacuate
- **Specific Needs:** Each has unique evacuation requirements (van, route guidance, group transport, etc.)
- **Less Dramatic:** Simpler conversation arcs focused on logistics rather than persuasion

## Testing the New Characters

### How to Test:

1. Start the backend server:
   ```bash
   cd /Users/tzhang/projects/A2I2/backend
   uvicorn server:app --host 0.0.0.0 --port 8001
   ```

2. Start the frontend:
   ```bash
   cd /Users/tzhang/projects/A2I2/frontend
   python -m http.server 8000
   ```

3. Visit: `http://localhost:8000`

4. Select any of the new characters from the grid

5. Try both modes:
   - **Interactive Mode:** Chat with the character as an operator
   - **Auto Mode:** Generate full conversation automatically

### Expected Behavior:

- **Mary:** Will mention her dog Poppy and need for van pickup
- **Ben:** Will ask about saving work and bringing pet gecko
- **Ana:** Will explain about seniors at the center needing help
- **Tom:** Will offer to help others and mention his truck
- **Mia:** Will be at school robotics lab, initially unaware

## Future Enhancements

### Possible Additions:

1. **Character-Specific Conversation Branches**
   - Like Bob and Michelle, could add more complex decision trees
   - Example: Tom's choice to help others vs. self-evacuation

2. **More Dialogue Variations**
   - Add more response options for each category
   - Include character-specific reactions to different operator approaches

3. **Visual Character Profiles**
   - Add character avatars/images to selection cards
   - Show character age, location, and needs more prominently

4. **Statistics Dashboard**
   - Track evacuation success rates per character
   - Show average conversation length
   - Display most effective persuasion techniques

## Technical Notes

### File Structure:
```
A2I2/
├── frontend/
│   └── index.html (✓ Updated - 10 characters)
├── backend/
│   └── server.py (✓ Updated - loads new character data)
└── data_for_train/
    ├── persona.json (✓ Already had data)
    ├── character_lines.jsonl (✓ Updated)
    └── characterlines.jsonl (✓ Updated)
```

### Dialogue Data Format:
Each character entry in JSONL files follows this structure:
```json
{
  "character": "name",
  "greetings": ["Hello?", "Hi", ...],
  "response_to_operator_greetings": [...],
  "progression": [...],
  "observations": [...],
  "general": [...],
  "closing": [...]
}
```

## Troubleshooting

### Character Not Appearing:
- Clear browser cache (Cmd/Ctrl + Shift + R)
- Check that both dialogue files are updated
- Restart backend server

### Character Not Responding:
- Verify dialogue data loaded: check server startup logs
- Confirm persona data exists for character (lowercase name)
- Check OpenAI API key is set in `.env` file

### Dialogue Seems Generic:
- Add more specific lines to character's dialogue data
- Enhance `response_to_operator_greetings` with character-specific details
- Consider adding character-specific conversation branches in server.py

## Contact

For issues or questions about the new characters, check:
- Main README.md for general setup
- OPENAI_SETUP.md for API configuration
- Backend logs for debugging information

