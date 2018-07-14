import paho.mqtt.client as mqtt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

style.use('fivethirtyeight')
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

dataFile = open("data.txt", "a");
Lkamall = np.array([])
Rkamall = np.array([])

def on_subscribe(client, obj, mid, granted_qos):
    print("Subscribed: " + str(obj) + " " + str(granted_qos));


def on_connect(client, userdata, flags, rc):
    print("rc: " + str(rc));
    # client.subscribe("accelerometer", 0);
    # client.subscribe("fusion", 0);
    # client.subscribe("gyroscope", 0);
    client.subscribe("kam", 0);


def on_message(client, userdata, msg):
    global Lkamall,Rkamall
    msg = (str(msg.payload))[2:-1]
    # print(msg)
    # kam = eval(str(msg))
    # lkam = kam["lkam"]
    # rkam = kam["rkam"]
    # Lkamall = np.concatenate((Lkamall, lkam))
    # Rkamall = np.concatenate((Rkamall, rkam))
    dataFile.write(msg)



mqttc = mqtt.Client(transport="websockets");
mqttc.on_connect = on_connect;
mqttc.on_subscribe = on_subscribe
mqttc.on_message = on_message
mqttc.connect("ec2-18-217-114-173.us-east-2.compute.amazonaws.com", 1883, 60);
mqttc.loop_forever()



