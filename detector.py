import cv2
import numpy as np
import time
from ultralytics import YOLO
from PIL import Image
from sklearn.cluster import DBSCAN

model = YOLO("./best.pt")  

cap = cv2.VideoCapture(0)  
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

window_name = "Blackjack Helper"
cv2.namedWindow(window_name)

def suitless(cls):
    """Remove suit from card class if present."""
    return cls.upper().rstrip('CDHS')

def rank_of(cls):
    """Get the readable rank name of a card."""
    ranks = {
        "2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six",
        "7": "Seven", "8": "Eight", "9": "Nine", "10": "Ten",
        "J": "Jack", "Q": "Queen", "K": "King", "A": "Ace"
    }
    return ranks.get(suitless(cls))

def pluralize(cls):
    """Pluralize a card rank name."""
    if cls == "Six":
        return "Sixes"
    return cls + "s"

def get_value_of_card(cls):
    """Get the numerical value of a card for blackjack."""
    rank = suitless(cls)
    if rank in ["J", "Q", "K", "10"]:
        return 10
    if rank == "A":
        return 11
    return int(rank) if rank.isdigit() else 0

def is_an_ace(cls):
    """Check if a card is an ace."""
    return suitless(cls) == "A"

def determine_action(player_cards, dealer_upcard, total, soft, is_pair, busted, is_blackjack):
    """
    Determine the best action for the current hand.
    
    Args:
        player_cards: List of player's cards
        dealer_upcard: Dealer's face-up card
        total: Total value of player's hand
        soft: Boolean indicating if hand is soft (contains an ace counted as 11)
        is_pair: Boolean indicating if hand is a pair
        busted: Boolean indicating if hand is busted (over 21)
        is_blackjack: Boolean indicating if hand is blackjack
        
    Returns:
        String with recommended action
    """
    if is_blackjack:
        return "You win"
    if busted:
        return "You lose"

    dealer_value = get_value_of_card(dealer_upcard)

    if len(player_cards) == 2:
        if is_pair:
            pair_of = suitless(player_cards[0]["class"])
            if pair_of in ["8", "A"]:
                return "SPLIT"
            if pair_of in ["2", "3", "6", "7", "9"] and dealer_value < 7:
                return "SPLIT"
        
        if soft:
            if 13 <= total <= 16 and dealer_upcard in ["5", "6"]:
                return "DOUBLE DOWN"
            if (total == 17 or total == 18) and dealer_value < 7:
                return "DOUBLE DOWN"
        else:
            if total == 9 and dealer_value < 7:
                return "DOUBLE DOWN"
            if total == 10 and dealer_value < 10:
                return "DOUBLE DOWN"
            if total == 11:
                return "DOUBLE DOWN"

    if soft:
        if total < 18:
            return "HIT"
        if total > 18:
            return "STAY"
        if dealer_value < 8:
            return "STAY"
        return "HIT"
    else:
        if total < 12:
            return "HIT"
        if dealer_value < 7:
            if total == 12 and dealer_upcard in ["2", "3"]:
                return "HIT"
            return "STAY"
        else:
            if total < 17:
                return "HIT"
            return "STAY"

def assign_cards_by_position(predictions):
    """
    Assign cards to player or dealer based on spatial clustering.
    
    Args:
        predictions: List of card predictions with bbox information
        
    Returns:
        Tuple of (player_cards, dealer_upcard)
    """
    if len(predictions) <= 1:
        return predictions, None
    
    card_positions = np.array([[p["bbox"]["x"], p["bbox"]["y"]] for p in predictions])
    
    clustering = DBSCAN(eps=200, min_samples=1).fit(card_positions)
    labels = clustering.labels_
    
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    if num_clusters <= 1:
        return predictions, None
    
    unique_labels, counts = np.unique(labels, return_counts=True)
    player_cluster = unique_labels[np.argmax(counts)]
    

    avg_distances = []
    for i, pos in enumerate(card_positions):
        distances = np.linalg.norm(card_positions - pos, axis=1)
        other_distances = distances[distances > 0]
        avg_distances.append(np.mean(other_distances) if len(other_distances) > 0 else float('inf'))
    

    candidate_indices = [i for i, label in enumerate(labels) if label != player_cluster]
    
    if not candidate_indices:
        dealer_idx = np.argmax(avg_distances)
    else:
        candidate_distances = [avg_distances[i] for i in candidate_indices]
        dealer_idx = candidate_indices[np.argmax(candidate_distances)]
    
    dealer_card = predictions[dealer_idx]
    player_cards = [p for i, p in enumerate(predictions) if i != dealer_idx]
    
    return player_cards, dealer_card

prev_time = time.time()
past_frame_times = []

COLOR_THEME = {
    "background": (40, 40, 40),
    "text": (255, 255, 255),
    "hit": (0, 255, 255),     # Yellow
    "stay": (0, 255, 0),      # Green
    "double": (255, 0, 255),  # Purple
    "split": (0, 165, 255),   # Orange
    "lose": (0, 0, 255),      # Red
    "win": (0, 255, 0),       # Green
    "ace": (0, 255, 255),     # Yellow
    "card": (0, 255, 0),      # Green
    "player_card": (0, 255, 0),  # Green
    "dealer_card": (255, 0, 0),  # Red
    "button": {
        "bg": (200, 200, 200),
        "selected_bg": (0, 200, 200),
        "border": (100, 100, 100),
        "selected_border": (0, 255, 255),
        "text": (0, 0, 0)
    }
}

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame")
        break

    overlay = frame.copy()
    
    results = model.predict(frame, conf=0.5, imgsz=640)

    predictions = []
    for result in results:
        boxes = result.boxes.xywh.cpu().numpy()  
        scores = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        for box, score, class_id in zip(boxes, scores, class_ids):
            predictions.append({
                "class": model.names[class_id],
                "confidence": float(score),
                "bbox": {
                    "x": float(box[0]),
                    "y": float(box[1]),
                    "width": float(box[2]),
                    "height": float(box[3])
                }
            })

    unique_cards = {}
    for p in predictions:
        if p["class"] not in unique_cards:
            unique_cards[p["class"]] = p
        else:
            if p["confidence"] > unique_cards[p["class"]]["confidence"]:
                unique_cards[p["class"]] = p

    cards = list(unique_cards.values())
    
    player_cards, dealer_card = assign_cards_by_position(cards)
    
    dealer_upcard = None
    if dealer_card:
        dealer_upcard = suitless(dealer_card["class"])
    
    soft = False
    is_pair = len(player_cards) == 2 and (len(player_cards) > 0 and 
                                         rank_of(player_cards[0]["class"]) == 
                                         rank_of(player_cards[1]["class"]))
    total = 0
    number_of_aces = 0
    
    for card in player_cards:
        value = get_value_of_card(card["class"])
        total += value
        if is_an_ace(card["class"]):
            number_of_aces += 1
            soft = True
    
    is_blackjack = len(player_cards) == 2 and total == 21
    busted = total > 21

    if busted and number_of_aces > 0:
        while number_of_aces > 0 and total > 21:
            total -= 10
            number_of_aces -= 1
        
        busted = total > 21
        soft = number_of_aces > 0

    action = "Place cards on camera" 
    if player_cards and dealer_upcard:
        action = determine_action(player_cards, dealer_upcard, total, soft, is_pair, busted, is_blackjack)

    info_panel_height = 140
    alpha = 0.7 
    cv2.rectangle(overlay, (0, 0), (frame_width, info_panel_height), COLOR_THEME["background"], -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    for i, pred in enumerate(cards):
        x, y, w, h = pred["bbox"]["x"], pred["bbox"]["y"], pred["bbox"]["width"], pred["bbox"]["height"]
        x1, y1 = int(x - w / 2), int(y - h / 2)
        x2, y2 = int(x + w / 2), int(y + h / 2)
        
        card_value = get_value_of_card(pred["class"])
        card_label = f"{pred['class']} ({card_value})"
        
        is_dealer = pred == dealer_card
        color = COLOR_THEME["dealer_card"] if is_dealer else COLOR_THEME["player_card"]
        
        if is_an_ace(pred["class"]):
            thickness = 5
        else:
            thickness = 3
            
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        role = "Dealer" if is_dealer else "Player"
        label = f"{role}: {card_label}"
        
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(frame, (x1, y1 - text_size[1] - 10), (x1 + text_size[0] + 10, y1), color, -1)
        cv2.putText(frame, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    current_time = time.time()
    if prev_time:
        past_frame_times.append(current_time - prev_time)
        if len(past_frame_times) > 30:
            past_frame_times.pop(0)
        total_fps = sum(past_frame_times)
        fps = len(past_frame_times) / total_fps if total_fps > 0 else 0
        cv2.putText(frame, f"FPS: {int(fps)}", (frame_width - 120, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_THEME["text"], 2)
    prev_time = current_time

    unique_card_count = len(cards)
    player_card_count = len(player_cards)
    cv2.putText(frame, f"Cards: {unique_card_count}", (frame_width - 220, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_THEME["text"], 2)
    cv2.putText(frame, f"Player: {player_card_count}", (frame_width - 220, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_THEME["text"], 2)
    cv2.putText(frame, f"Dealer: {1 if dealer_card else 0}", (frame_width - 220, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_THEME["text"], 2)

    hand_state = "Find a Blackjack Hand" if not player_cards else (
        f"BUST ({total})" if busted else
        "BLACKJACK!" if is_blackjack else
        f"Pair of {pluralize(rank_of(player_cards[0]['class']))}" if is_pair else
        f"{'Soft' if soft else 'Hard'} {total}"
    )

    dealer_text = f"Dealer Shows: {dealer_upcard}" if dealer_upcard else "No dealer card detected"
    cv2.putText(frame, dealer_text, (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_THEME["text"], 2)
    cv2.putText(frame, hand_state, (20, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_THEME["text"], 2)
    
    action_color = COLOR_THEME["text"]
    if action == "HIT":
        action_color = COLOR_THEME["hit"]
    elif action == "STAY":
        action_color = COLOR_THEME["stay"]
    elif action == "DOUBLE DOWN":
        action_color = COLOR_THEME["double"]
    elif action == "SPLIT":
        action_color = COLOR_THEME["split"]
    elif "lose" in action.lower():
        action_color = COLOR_THEME["lose"]
    elif "win" in action.lower():
        action_color = COLOR_THEME["win"]
        
    text_size = cv2.getTextSize(action, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
    cv2.rectangle(frame, (20, 95), (30 + text_size[0], 95 + text_size[1] + 10), (0, 0, 0), -1)
    cv2.putText(frame, action, (25, 125), cv2.FONT_HERSHEY_SIMPLEX, 1.5, action_color, 3)

    instructions = "Press 'q' to quit | Cards separated by distance are auto-assigned"
    cv2.putText(frame, instructions, (frame_width // 2 - 280, frame_height - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_THEME["text"], 2)

    cv2.imshow(window_name, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()