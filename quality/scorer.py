# quality/scorer.py

def evaluate_response(prompt: str, response_text: str) -> float:
    """
    Yeh function check karta hai ki AI ka response prompt ke hisab se 
    kitna accha hai (0.0 se 1.0 ke beech score deta hai).
    """
    score = 1.0
    
    # Rule 1: Agar response bohot chota hai, toh score kam kar do
    if len(response_text) < 10:
        score -= 0.5
        
    # Rule 2: Agar prompt mein 'error' ya 'fail' jaisa kuch hai
    if "error" in response_text.lower():
        score -= 0.4
        
    # Score ko 0.0 aur 1.0 ke beech rakhein
    return max(0.0, min(1.0, score))

# Test karne ke liye chota sa code
if __name__ == "__main__":
    test_prompt = "What is MLOps?"
    test_response = "MLOps is machine learning operations."
    
    final_score = evaluate_response(test_prompt, test_response)
    print(f"Quality Score for response: {final_score}")