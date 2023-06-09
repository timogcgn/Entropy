from Samplerz import sample_gaussian

def approx_falcon_gaussian(n,key_samples):
    # this function samples key_samples many vectors of length n from D(1.17*sqrt(2*12289/n)) and derives a probability distribution
    dist={}
    for i in range(key_samples):
        sample=sample_gaussian(n)
        for val in sample:
            dist[abs(val)]=dist.get(abs(val),0)+1/(n*key_samples)
    keylist1=[]
    for key in dist:
        keylist1+=[key]
    keylist1.sort(reverse=True)
    if keylist1[len(keylist1)-1]==0:
        keylist1=keylist1[:len(keylist1)-1]
    keylist2=keylist1.copy()
    keylist2.sort()
    out={}
    for key in keylist1:
        out[-key]=dist[key]/2
    if 0 in dist:
        out[0]=dist[0]
    for key in keylist2:
        out[key]=out[-key]
    return out

falcon512dist={-20: 1/1048576,
 -19: 1/524288,
 -18: 3/1048576,
 -17: 19/1048576,
 -16: 43/1048576,
 -15: 117/1048576,
 -14: 65/262144,
 -13: 19/32768,
 -12: 1323/1048576,
 -11: 2613/1048576,
 -10: 4887/1048576,
 -9: 4359/524288,
 -8: 7321/524288,
 -7: 23297/1048576,
 -6: 34803/1048576,
 -5: 187/4096,
 -4: 63141/1048576,
 -3: 609/8192,
 -2: 45531/524288,
 -1: 25219/262144,
 0: 52049/524288,
 1: 25219/262144,
 2: 45531/524288,
 3: 609/8192,
 4: 63141/1048576,
 5: 187/4096,
 6: 34803/1048576,
 7: 23297/1048576,
 8: 7321/524288,
 9: 4359/524288,
 10: 4887/1048576,
 11: 2613/1048576,
 12: 1323/1048576,
 13: 19/32768,
 14: 65/262144,
 15: 117/1048576,
 16: 43/1048576,
 17: 19/1048576,
 18: 3/1048576,
 19: 1/524288,
 20: 1/1048576}

falcon1024dist={-13: 3/1048576,
 -12: 11/524288,
 -11: 109/1048576,
 -10: 307/1048576,
 -9: 1055/1048576,
 -8: 1465/524288,
 -7: 3711/524288,
 -6: 8099/524288,
 -5: 3983/131072,
 -4: 55367/1048576,
 -3: 84393/1048576,
 -2: 113981/1048576,
 -1: 2151/16384,
 0: 72973/524288,
 1: 2151/16384,
 2: 113981/1048576,
 3: 84393/1048576,
 4: 55367/1048576,
 5: 3983/131072,
 6: 8099/524288,
 7: 3711/524288,
 8: 1465/524288,
 9: 1055/1048576,
 10: 307/1048576,
 11: 109/1048576,
 12: 11/524288,
 13: 3/1048576}