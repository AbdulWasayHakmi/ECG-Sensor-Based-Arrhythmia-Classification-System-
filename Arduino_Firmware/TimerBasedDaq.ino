bool flag = 0;
bool start = 0;
int readVal = 0;
void setup() {
  // put your setup code here, to run once:
  pinMode(A0,INPUT);
  Serial.begin(9600);
  cli();
  TCCR1A = 0;
  TCCR1B = 0;
  TCCR1B |= 0b100;
  TCCR1B |= (1 << WGM12);
  TIMSK1 |= 0b10;0
  OCR1A = 500;
  sei();
}

void loop() {
  // put your main code here, to run repeatedly:
  if(!start)
  {
    Serial.println('\r');
    Serial.println('\r');
    Serial.println('\r');
    Serial.println('\r');
    start = 1;
  }
  if(flag)
  {
    flag = 0;
    char buffer[5];         

    sprintf(buffer, "%04d", readVal); 
    Serial.print(buffer);
  }
}

ISR(TIMER1_COMPA_vect)
{
  readVal = analogRead(A0);
  flag = 1;
}

