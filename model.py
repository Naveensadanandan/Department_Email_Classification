from util import prediction_pipe
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class Email(BaseModel):
    """
    Pydantic model representing the input email content for department prediction.
    
    Attributes:
        content (str): The content of the email to be classified.
    """
    content: str


@app.post("/predict-department/")
async def predict_department(email: Email):
    """
    Predicts the department of an email using a pre-trained BERT model.

    Args:
        email (Email): The email content wrapped in a Pydantic model.

    Returns:
        dict: A dictionary containing the predicted department.

    Raises:
        HTTPException: If there is any exception during the prediction process, 
                       it raises a 500 Internal Server Error.
    """
    text = f"{email.content}"
    
    try:
        department = prediction_pipe(text)
        return {"department": department}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
