from flask import Flask, jsonify, request
from flask_cors import CORS

# Enable CORS for all routes

app = Flask(__name__)
CORS(app)


# Sample data for bus numbers and locations
bus_numbers = ['101', '102', '103']
locations = [
    {'id': 1, 'name': 'Location 1'},
    {'id': 2, 'name': 'Location 2'},
    {'id': 3, 'name': 'Location 3'},
    {'id': 4, 'name': 'Location 4'},
    {'id': 5, 'name': 'Location 5'},
]

@app.route('/api/busNumbers', methods=['GET'])
def get_bus_numbers():
    return jsonify({'busNumbers': bus_numbers})

@app.route('/api/locations', methods=['GET'])
def get_locations():
    return jsonify({'locations': locations})

@app.route('/api/route', methods=['GET'])
def get_route():
    bus_no = request.args.get('busNo')
    start = request.args.get('start')
    stop = request.args.get('stop')
    
    if not bus_no or not start or not stop:
        return jsonify({'error': 'Missing parameters'}), 400
    
    # You could implement more advanced validation and logic here
    return jsonify({
        'message': f'Route selected: Bus {bus_no} from Location {start} to Location {stop}'
    })

if __name__ == '__main__':
    app.run(debug=True)
