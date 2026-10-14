# watchdog/drift.py

class DriftWatchdog:
    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold
        self.recent_scores = []

    def add_score(self, score: float):
        """Naye quality scores ko record karta hai"""
        self.recent_scores.append(score)
        # Hum sirf pichle 10 scores ko yaad rakhenge
        if len(self.recent_scores) > 10:
            self.recent_scores.pop(0)

    def check_drift(self) -> bool:
        """Check karta hai ki average quality score threshold se kam ho gaya hai kya"""
        if not self.recent_scores:
            return False
        
        avg_score = sum(self.recent_scores) / len(self.recent_scores)
        
        # Agar average score threshold se kam hai, toh drift detected!
        if avg_score < self.threshold:
            return True
        return False

# Test karne ke liye chota sa code
if __name__ == "__main__":
    watchdog = DriftWatchdog(threshold=0.6)
    
    # Kuch dummy scores pass karte hain
    scores = [0.9, 0.8, 0.4, 0.3, 0.2]
    for s in scores:
        watchdog.add_score(s)
        print(f"Added score: {s}, Drift Alert?: {watchdog.check_drift()}")