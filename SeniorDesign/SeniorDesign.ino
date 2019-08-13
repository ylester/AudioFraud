#include <Wire.h> //library for i2c communication

#include <BlinkM_funcs.h> //library for blinkLED functions

#include <SerLCD.h>
SerLCD lcd; // initialize the LCD with default i2c address on 0x72

byte blinkAddress = 0x09, LCDAddress = 0x72, red ,green, blue ; //initialize I2C address for LCD and LED and color variables
int index = -1;

String serialData=""; //empty string

String speaker, result;

 
void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);  // initialize serial for debugging
  Serial.println("Serial Initilized");

  Wire.begin(); // intialize i2c address
  Serial.println("I2C Initilized");

  BlinkM_off(blinkAddress); // turn off the LED at first
  BlinkM_setFadeSpeed(blinkAddress, 0xAF); // set fade speed for LED

  lcd.begin(Wire); // set up LCD for I2C communication
  lcd.setBacklight(255,255,255); // set up full brightness on LCD
  lcd.setContrast(0); // set contrast ( lower to 0 for higher contrast)
  lcd.clear(); // clear teh LCD and move cursor to home position
  Serial.println("Program Started");
  randomSeed(analogRead(0));
  delay(500); //pause for half a second
}

void loop() {
  // put your main code here, to run repeatedly:
  if (Serial.available()>0)  // if receivin gserial data  
  {
      serialData = Serial.readString(); //store it in a string
      lcd.clear(); // clear lcd screen
      // Serial.println (serialData); // print the serial datta on terminal for debgging 
      speaker = serialData.substring(0, serialData.indexOf(",")); // parse speaker
      result = serialData.substring((serialData.indexOf(",")+2),serialData.length()); // parse result
      lcd.print(speaker); // print speaker
      lcd.setCursor(0,1); //lcd.setCursor(column,row); set to second row
      lcd.print(result);  // print result on second line
      //Serial.println(result);
      BlinkM_fadeToRGB(blinkAddress,red,green,blue); // bink led in a RGB color 
      delay(1000);
    }
   else  //if no data available 
   {
      lcd.clear();
      lcd.print("No Data"); // print no data 
      BlinkM_off(blinkAddress); // turn off led
      delay(200);
   }
  

}
