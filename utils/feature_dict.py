import json
import os

def save_features(categories, object_names, features, epoch, acc, args):

    feature_dict = {}
    for i, name in enumerate(object_names):
        feature_dict[name] = {
            'category': categories[i],
            'epoch': epoch,
            'accuracy': acc,
            'feature': features[i].tolist()  # convert to list for JSON serialization
        }

    # Save to JSON
    json_path = os.path.join(args.experiment_path, f"features/ESB_features_{epoch}.json")
    os.makedirs(os.path.dirname(json_path), exist_ok=True)  # Create directory if it doesn't exist
    with open(json_path, 'w') as f:
        json.dump(feature_dict, f, indent=2)

    return json_path
    