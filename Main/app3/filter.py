import json
import os
import random
import time
from datetime import datetime

ALERTS_FILE = os.path.join(os.path.dirname(__file__), "alerts.json")

def process_detections(boxes):
    """
    Checks if a major flood is detected.
    Saves an alert if detected, ensuring we don't spam alerts too frequently.
    """
    if not boxes or len(boxes) == 0:
        return
    
    alerts = []
    if os.path.exists(ALERTS_FILE):
        try:
            with open(ALERTS_FILE, "r") as f:
                alerts = json.load(f)
        except Exception:
            pass
            
    if alerts:
        last_alert = alerts[-1]
        last_time = datetime.fromisoformat(last_alert["timestamp"])
        if (datetime.now() - last_time).total_seconds() < 10:
            return
            
    # Dummy location data for testing/demo
    lat = round(random.uniform(26.0, 27.0), 4) # North Bengal region roughly
    lon = round(random.uniform(88.0, 89.0), 4)
    
    alert = {
        "timestamp": datetime.now().isoformat(),
        "message": f"🚨 ALERT: Potential major flood area detected.\nLocation: Lat {lat}, Lon {lon}.\nImmediate attention recommended.",
        "boxes_count": len(boxes),
        "id": int(time.time() * 1000)
    }
    
    alerts.append(alert)
    
    # Keep only the last 100 alerts
    if len(alerts) > 100:
        alerts = alerts[-100:]
    
    with open(ALERTS_FILE, "w") as f:
        json.dump(alerts, f, indent=2)

def get_recent_alerts(since_id=0):
    """
    Get alerts that have an ID greater than since_id.
    """
    if not os.path.exists(ALERTS_FILE):
        return []
    try:
        with open(ALERTS_FILE, "r") as f:
            alerts = json.load(f)
            if since_id:
                return [a for a in alerts if a["id"] > int(since_id)]
            return alerts
    except Exception:
        return []

def clear_alerts():
    """Clear all alerts."""
    if os.path.exists(ALERTS_FILE):
        try:
            os.remove(ALERTS_FILE)
        except Exception:
            pass

clear_alerts()
