x = linspace(0,8,1000);
f1 = sin(x.^2)+cos(x./2);
f2 = cos(x)+sin(x);
plot(x,f1,'--m');
hold on;
plot(x,f2,'-ob');
hold off;




