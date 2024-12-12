/*
 Hardware Connections (Breakoutboard to Arduino):
  -5V = 5V (3.3V is allowed)
  -GND = GND
  -SDA = A4 (or SDA)
  -SCL = A5 (or SCL)
  -INT = Not connected

  The MAX30105 Breakout can handle 5V or 3.3V I2C logic. We recommend powering the board with 5V
  but it will also run at 3.3V.
  */


#include <WiFi.h>
#include <HTTPClient.h>
#include <Wire.h>
#include "MAX30105.h"
#include "heartRate.h"

// https://www.upesy.com/blogs/tutorials/how-to-connect-wifi-acces-point-with-esp32#connecting-to-a-wi-fi-access-point


#define MAX_BRIGHTNESS 255

// Define GSR pin
#define GSR_PIN A0 // Analog pin for GSR sensor

const char* ssid = "CommunityFibre10Gb_98A69";        // Replace with your WiFi SSID
const char* password = "u6e6pdbus2"; // Replace with your WiFi Password

//const char* ssid = "Alexis Almeida's Iphone 13";        // Replace with your WiFi SSID
//const char* password = "suki1234"; // Replace with your WiFi Password//

const char* serverName = "http://192.168.1.232:5002/receive_data"; 
//const char* serverName = "http://192.168.1.232:5002/send_data"; //streamlit




MAX30105 particleSensor;

const byte RATE_SIZE = 4;
byte rates[RATE_SIZE];
byte rateSpot = 0;
long lastBeat = 0;
float beatsPerMinute = 0;
int beatAvg;


String get_wifi_status(int status){
    switch(status){
        case WL_IDLE_STATUS:
            return "WL_IDLE_STATUS";
        case WL_SCAN_COMPLETED:
            return "WL_SCAN_COMPLETED";
        case WL_NO_SSID_AVAIL:
            return "WL_NO_SSID_AVAIL";
        case WL_CONNECT_FAILED:
            return "WL_CONNECT_FAILED";
        case WL_CONNECTION_LOST:
            return "WL_CONNECTION_LOST";
        case WL_CONNECTED:
            return "WL_CONNECTED";
        case WL_DISCONNECTED:
            return "WL_DISCONNECTED";
        default:
            return "UNKNOWN_STATUS";
    }
}

void setup() {
    Serial.begin(115200);
    //debug.begin(115200);

    Serial.println("\nConnecting to WiFi");
    Serial.println(get_wifi_status(WiFi.status()));
    WiFi.begin(ssid, password);

    while (WiFi.status() != WL_CONNECTED) {
      
        Serial.println(get_wifi_status(WiFi.status()));
    }

    Serial.println("\nConnected to the WiFi network");
    Serial.print("Local ESP32 IP: ");
    Serial.println(WiFi.localIP());

    Serial.println("Initializing MAX30105 sensor...");
    if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
        Serial.println("MAX30105 was not found. Please check wiring/power.");
        while (1);
    }

    particleSensor.setup();
    particleSensor.setPulseAmplitudeRed(0x0A); //Turn Red LED to low to indicate sensor is running
    particleSensor.setPulseAmplitudeGreen(0); //Turn off Green LED
    Serial.println("MAX30105 sensor initialized.");

    //Temperature 
    particleSensor.enableDIETEMPRDY();

      // GSR pin setup
    pinMode(GSR_PIN, INPUT);
}


void send_packet(String jsonPayload){
    if(WiFi.status() == WL_CONNECTED) {
        HTTPClient http;

        http.begin(serverName);

        http.addHeader("Content-Type", "application/json");
        float temperature = particleSensor.readTemperature();
        
        jsonPayload += "\"ST\":" + String(temperature) +",";
        

          // Read raw GSR value and convert 
         int gsrValue = analogRead(GSR_PIN);
        
        jsonPayload += "\"GSR\":" + String(gsrValue);
        jsonPayload += "}";

        Serial.print("Sending data: ");
        Serial.println(jsonPayload);

        int httpResponseCode = http.POST(jsonPayload);

        if (httpResponseCode > 0) {
            String response = http.getString();
            Serial.print("HTTP Response code: ");
            Serial.println(httpResponseCode);
            Serial.print("Response from server: ");
            Serial.println(response);
        }
        else {
            Serial.print("Error on sending POST: ");
            Serial.println(httpResponseCode);
        }
        http.end();
    }
    else {
        Serial.println("WiFi Disconnected");
    }
}
unsigned long lastSend = 0;
const unsigned long interval = 10;

void loop() {
          long irValue = particleSensor.getIR();

 if (checkForBeat(irValue) == true)
  {
    if (irValue < 30000) Serial.println("No finger detected!");
    else {

  
    //We sensed a beat!
    long delta = millis() - lastBeat;
    lastBeat = millis();

    beatsPerMinute = 60 / (delta / 1000.0);

    if (beatsPerMinute < 255 && beatsPerMinute > 20)
    {
      rates[rateSpot++] = (byte)beatsPerMinute; //Store this reading in the array
      rateSpot %= RATE_SIZE; //Wrap variable

      //Take average of readings
      beatAvg = 0;
      for (byte x = 0 ; x < RATE_SIZE ; x++)
        beatAvg += rates[x];
      beatAvg /= RATE_SIZE;
    }
  }
  }
   
        String jsonPayload = "{";
        jsonPayload += "\"ir\": " + String(irValue) + ",";
        jsonPayload += "\"beatAvg\": " + String(beatAvg) + ",";
        jsonPayload += "\"HR\": " + String(beatsPerMinute)+ ",";


        if (lastBeat - lastSend >= interval){
          send_packet(jsonPayload);
          lastSend = lastBeat;
        }
        
    
    }
   

