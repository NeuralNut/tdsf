import torch
from torch.autograd import Variable, Parameter
## PDE as loss function. Thus would use the network which we call as u_theta
def f_partial(data, net, MaxX, MinX, MaxY, MinY, cnsts):

    
    xmax = torch.from_numpy(MaxX).to(device)
    xmin = torch.from_numpy(MinX).to(device)
    ymax = torch.from_numpy(MaxY).to(device)
    ymin = torch.from_numpy(MinY).to(device)
    
    wvec = Variable(data[:,:,0], requires_grad=True).to(device)
    hvec = Variable(data[:,:,1], requires_grad=True).to(device)
    tvec = Variable(data[:,:,2], requires_grad=True).to(device)

    s = net((torch.stack([wvec, hvec, tvec], -1) - xmin)/(xmax - xmin))
    
    # remove normalization
    s = s*(ymax-ymin) + ymin
    
    s_w = torch.autograd.grad(s.sum(), wvec, create_graph=True)[0]
    s_h = torch.autograd.grad(s.sum(), hvec, create_graph=True)[0] 
        
    ww = torch.reshape(data[:,:,0], (-1,1))
    hh = torch.reshape(data[:,:,1], (-1,1))
    tt = torch.reshape(data[:,:,2], (-1,1))
    
    # define parameters
    # TODO: define  params as input
    # ho = cnsts[:, 0].mean()
    # Ro = cnsts[:, 1].mean()
    # go = cnsts[:, 2].mean()
    # P1 = cnsts[:, 3].mean()
    # P2 = cnsts[:, 4].mean()
    # n = cnsts[:, 5].mean()
    # pv = cnsts[:, 6].mean()
    # sv = cnsts[:, 7].mean()
    # rho = cnsts[:, 8].mean()

    ho = 122
    Ro = 202
    go = 26
    P1 = 3.6*1e6
    P2 = 5.0*1e6
    n = 2.4
    pv = 3500
    sv = 2021
    rho = 2000
    
    dsdR = torch.zeros(ww.shape).to(device)

    Rel = Ro*((ho/hh)**(1/n))*(ww**(1/3))
    ga = go*Ro/Rel

    mu = rho*(sv**2)
    lam = rho*(pv**2)-2*mu    
    wo = pv/Rel
    bet = (lam+2*mu)/(4*mu)
    alp = wo/(2*bet)
    p = wo*(1/2/bet-1/4/bet**2)**(1/2)

    def dF_(tau):
        return (Rel*pv**2)/(4*mu*bet*p)*(-alp*torch.exp(-alp*tau)*torch.sin(p*tau) + p*torch.exp(-alp*tau)*torch.cos(p*tau))

    def dBdR_(tau):
        return (tt-tau)*ga*torch.exp(-ga*(tt-tau))/Rel*P1*(hh/ho) - ((tt-tau)*ga*torch.exp(-ga*(tt-tau))+3*(1-torch.exp(-ga*(tt-tau))))/Rel*P2*((ho/hh)**(1/3))*((Ro/Rel)**3)*(ww**(0.87))

    dtau = 0.001
    tau = torch.tensor(np.arange(0,1,dtau)).to(device).float()
    
    dF = dF_(tau)
    dBdR = dBdR_(tau)
    
    temp = dF*dBdR*dtau
    mask = (tt - tau) < 0
    temp[mask] = 0
    dsdR += temp.sum(axis=-1).unsqueeze(-1)

    dRdw = 1/3*Ro*((ho/hh)**(1/n))*(ww**(-2/3))
    dRdh = -1/n*((ho/hh)**(1/n))*(1/hh)*(ww**(1/3))
    dsdw = dsdR*dRdw
    dsdh = dsdR*dRdh
       
        
    return s_w, dsdw, s_h, dsdh