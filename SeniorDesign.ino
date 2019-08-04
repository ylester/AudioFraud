#include <Wire.h> //library for i2c communication

#include <BlinkM_funcs.h> //library for blinkLED functions

#include <SerLCD.h>
SerLCD lcd; // initialize the LCD with default i2c address on 0x72

byte blinkAddress = 0x09, LCDAddress = 0x72, red ,green, blue ; //initialize I2C address for LCD and LED and color variables

void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);  // initialize serial for debugging
  Serial.println("Serial Initilized");

  Wire.begin(); // intialize i2c address
  Serial.println("I2C Initilized");

  BlinkM_off(blinkAddress); // turn off the LED at first
  BlinkM_setFadeSpeed(blinkAddress, 0x05); // set fade speed for LED

  lcd.begin(Wire); // set up LCD for I2C communication
  lcd.setBacklight(255,255,255); // set up full brightness on LCD
  lcd.setContrast(0); // set contrast ( lower to 0 for higher contrast)
  lcd.clear(); // clear teh LCD and move cursor to home position
  lcd.print("Program Started");
  randomSeed(analogRead(0));
  delay(500); //pause for half a second
}

void loop() {
  // put your main code here, to run repeatedly:
    lcd.clear(); //Clear the display - this moves the cursor to home position as well
    lcd.print("Rec. Started");
    lcd.setCursor(0, 1);
    lcd.print(millis() / 1000);

    red = byte(random(255));
    green = byte(random(255));
    blue = byte(random(255));
    BlinkM_setRGB(blinkAddress, red , green, blue);
    delay(400);     

}
