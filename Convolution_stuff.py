# the convolution function is helpful for finding the distribution family B(eta) by utilizing convolution(B(eta),B(zeta))=B(eta+zeta).

def convolution(D1, D2):

    out = {}
    for key_1 in D1:
        for key_2 in D2:
            key = key_1+key_2
            out[key] = out.get(key, 0) + D1[key_1] * D2[key_2]
    return out

def self_convolution(D,i):
    out={0:1}
    ex=bin(i)[2:]
    for j in range(len(ex)):
        out=convolution(out,out)
        if ex[j]=='1':
            out=convolution(out,D)
    return out

def multiplicity(D):
    D_m={}
    for key in D:
        p=D[key]
        D_m[p]=D_m.get(p,0)+1
    del(p)
    P=[]
    for key in D_m:
        P+=[key]
    P.sort(reverse=True)
    m=[]
    for p in P:
        m+=[D_m[p]]
    return P,m