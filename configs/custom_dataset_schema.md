# Custom Dataset Format

Your CSV must have at minimum these two columns:

| question | expected_answer |
|---|---|
| What is our refund policy? | Refunds are processed within 14 days |
| How do I reset my password? | Click forgot password on the login page |

Optional columns that enable richer evaluation:

| Column | Purpose |
|---|---|
| id | Unique identifier (auto-generated if missing) |
| subject | Category for grouped analysis (e.g. "billing", "technical") |
| choices | Pipe-separated MCQ options: "A. Yes\|B. No\|C. Maybe" |

Example with all columns:
question,expected_answer,subject,id
"What is 2+2?","4","math","q001"
"Capital of France?","Paris","geography","q002"