function [C Ceq] = AiC_GA_nonlinear_constraints_INFILLED(Input)
%rng(15)
x1 = double(uint8(Input(1))); % Number of Storeys
x2= double(uint8(Input(2))); % Number of Spans
x3 = Input(3); % Length of Spans
x4 = double(uint8(Input(4))); % Opening percentage
x5 = Input(5); % Masonry wall stiffness
x6 = Input(6); % Column width a
x7 = Input(7); % Fundamental period from ANN

I = (x6)^4/12; %Moment inercije na osnovu dimenzije stuba
Lw = x3-x6;
alphaw = (Lw*x2*2.4*x1*(x4/100))/(Lw*x2*2.4*x1);
lambdah = 3*((x5*(10^5)*sin(2*(atan(2.4/Lw))))/(4*31*(10^6)*I*2.4))^(1/4);

%INEQUALITY CONSTRAINTS <=0 SHAPE
C(1) = -2*(alphaw)^(0.54)+(alphaw)^(1.14); % INEQUALITY 2
C(2) = 0.5 - x7; % INEQUALITY 2
C(3) = sqrt((Lw)^2+2.4^2)*0.175*(lambdah)^(-0.4)*(1-2*(alphaw)^(0.54)+(alphaw)^(1.14))-1.23; % INEQUALITY 3
C(4) = lambdah-4.82; % INEQUALITY 4
C(5) = -lambdah+1.02; % INEQUALITY 5
C(6) = -Lw+2.15; % INEQUALITY 6
C(7) = Lw-7.15; % INEQUALITY 7
C(8) = alphaw-0.75;
C(9) = -alphaw+0;

Ceq=[]; %NO EQUALITY CONSTRAINTS