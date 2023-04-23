function MOO = AiC_GA_multiobjective_INFILLED(Input)
%rng(15)
x1 = double(uint8(Input(1))); % Number of Storeys
x2= double(uint8(Input(2))); % Number of Spans
x3 = Input(3); % Length of Spans
x4 = double(uint8(Input(4))); % Opening percentage
x5 = Input(5); % Masonry wall stiffness
x6 = Input(6); % Column width a
x7 = Input(7); % Fundamental period from ANN

I = (x6)^4/12; 
Lw = x3-x6;
alphaw = (Lw*x2*2.4*x1*(x4/100))/(Lw*x2*2.4*x1);
lambdah = 3*((x5*(10^5)*sin(2*(atan(2.4/Lw))))/(4*31*(10^6)*I*2.4))^(1/4);

f1= x7; %FUNDAMENTAL PERIOD
f2=lambdah; %RELATIVE PANEL TO FRAME STIFFNESS

MOO = [f1 f2];