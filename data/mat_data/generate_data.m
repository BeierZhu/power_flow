mpc = loadcase(case14_beier);
result = runopf(case14_beier);
V = result.bus(:,8);
theta = result.bus(:,9);
bus = mpc.bus;
Y = makeYbus(mpc);
Y = full(Y);
gen = mpc.gen;
baseMVA = mpc.baseMVA;
save('case14.0.25.mat','bus','gen','Y','baseMVA','V','theta')