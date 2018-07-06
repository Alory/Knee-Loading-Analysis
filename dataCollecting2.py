import paho.mqtt.client as mqtt
import json
import time
import os

dataFile = open("data.txt","a");
recording = False;
recordFile = None;
print(recording);

def on_connect(client, userdata, flags, rc):
    print("rc: " + str(rc));
    # client.subscribe("accelerometer", 0);
    # client.subscribe("fusion", 0);
    # client.subscribe("gyroscope", 0);
    client.subscribe("recordFlag", 0);
    client.subscribe("data", 0);
def on_message(client, obj, msg):
	global recording,dataFile,recordFile;
	print("topic : " + msg.topic)
	if(msg.topic == "recordFlag"):
		print("recordFlag : " + str(msg.payload));
		timeNow = time.strftime('%Y-%m-%d %H-%M-%S',time.localtime(time.time()));
		if(str(msg.payload) == "sensorsOn"):
			recordFile = open(timeNow + ".txt","a");
			recording = True;
		else:
			recording = False;
			recordFile.close();
			dataFile.close();
	else:
		print(str(msg.payload));
		if(recording == True):
			# payloadJson = json.loads(str(msg.payload));
			# recordFile.write(msg.topic + " : " + str(msg.payload) + "\n");
			recordFile.write(str(msg.payload))
		else:
			try:
				dataFile = open("data.txt","a");
			except Exception as e:
				pass
			dataFile.write(msg.topic + " : " + str(msg.payload) + "\n");

		
def on_subscribe(client, obj, mid, granted_qos):
    print("Subscribed: " + str(obj) + " " + str(granted_qos));

mqttc = mqtt.Client(transport="websockets");
mqttc.on_connect = on_connect;
mqttc.on_message = on_message
mqttc.on_subscribe = on_subscribe
mqttc.connect("ec2-18-217-114-173.us-east-2.compute.amazonaws.com",1883,60);
# rc=0;
# while rc == 0:
#     rc = mqttc.loop()
# print("rc: " + str(rc))
mqttc.loop_forever()

# import paho.mqtt.client as mqtt
    
# # The callback for when the client receives a CONNACK response from the server.
# def on_connect(client, userdata, flags, rc):
#     print("Connected with result code "+str(rc))

#     # Subscribing in on_connect() means that if we lose the connection and
#     # reconnect then subscriptions will be renewed.
#     client.subscribe("test")

# # The callback for when a PUBLISH message is received from the server.
# def on_message(client, userdata, msg):
#     print(msg.topic+" "+str(msg.payload))

# client = mqtt.Client(client_id="", clean_session=True, userdata=None,
# 	protocol=mqtt.MQTTv311,transport="websockets");
# client.on_connect = on_connect
# client.on_message = on_message

# client.connect("ec2-18-217-114-173.us-east-2.compute.amazonaws.com", 1883, 60)

# rc = 0
# while rc == 0:
#     rc = client.loop()
# print("rc: " + str(rc))
# #client.loop_forever()  
