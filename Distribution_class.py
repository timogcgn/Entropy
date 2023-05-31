import gc

from Entropy_stuff import *
from Multinomial import *
from Largelog import *
from Root_sum import approx_sum_of_roots
from Compact_Dictionary import *

class distribution:
    def __init__(self, dist, base=2):
        cumu=0
        for key in dist:
            cumu+=dist[key]
        if cumu==0:
            print("Not a valid distribution")
        elif cumu!=1:
            self.d = {}
            for key in dist:
                self.d[key]=dist[key]/cumu
            self.dist=self.d
        else:
            self.d = dist
            self.dist = dist
        self.p, self.m=multiplicity(self.dist)
        self.label={}
        for prob in self.p:
            self.label[prob]=[]
            for key in self.dist:
                if self.dist[key]==prob:
                    self.label[prob]+=[key]
        eta=len(self.p)-1
        self.eta = eta
        self.range = list(self.d)
        self.log_p = []
        for prob in self.p:
            self.log_p += [-largelog(prob,2)]
        self.entropy = dist_entropy(self.dist, 1, base=base)
        self.comp_dics={}
        
    def __repr__(self, indent=10):
        l=str(indent)
        out="Distribution with sampling probabilities:\n"
        for key in self.label:
            b=str(self.label[key][0])
            for i in range(1,len(self.label[key])):
                b+=", " + str(self.label[key][i])
            out+='{0: >10}'.format(b) + " : " + str(key) + "\n"
        out+="\n"
        out+="Entropy of distribution is " + str(self.entropy)
        return out
    
    
    ### PROBABILITY FUNCTIONS ###
    
    def prob(self, i, f=False):
        # returns the probability of sampling i.
        if f:
            return float(self.d.get(i,0))
        return self.d.get(i,0)
    
    def vec_prob(self, v, f=False):
        # returns the probability of sampling a specific vector v=[i_1,...,i_n] from A^n.
        p=1
        for i in v:
            p*=self.prob(i, f=f)
        return p
        
    def compact_vec_prob(self, l, f=False):
        # returns the probability of sampling a specific vector with unsigned weight l=[n_0,...,n_eta]
        # (e.g. self.vec_prob([0,1,-1]) = self.compact_vec_prob([1,2]) ).
        p=0
        if len(l)-1<=self.eta:
            p=1
            for i in range(len(l)):
                p*=self.prob(i, f=f)**l[i]
        return p
    
    
    ### COMPACT DICTIONARY FUNCTIONS ###
    
    def comp_dic_list(self, n):
        # returns the compact dictionary for vectors of length n if it already exists; 
        # otherwise, returns an empty compact dictionary.
        return self.comp_dics.get(n,empty_comp_dic)
    
    def create_par_comp_dic(self, n, p_limit,cur_j,high_n,L,p, if_print=True):
        # iterative function for par_comp_dic and comp_dic.
        
        # iterates over all viable vector weights with sufficient sampling probability, gathers them in a list with
        # their sampling probability and their amount of occurrences(i.e. the amount of unsigned permutations)
        # and sorts that list by probability.
        if cur_j==self.eta and self.log_p[self.eta]*high_n<=p_limit:
            L[cur_j]=high_n
            if if_print:
                print(L)
            count=count_with_multiplicities(L, self.m)
            self.comp_dics[n].dic+=[[L.copy(),p*self.p[cur_j]**high_n,count]]
            self.comp_dics[n].count+=count
            self.comp_dics[n].p+=count*p*self.p[cur_j]**high_n
            L[cur_j]='-'
        elif high_n==0:
            for j in range(cur_j, self.eta+1):
                L[j]=0
            if if_print:
                print(L)
            count=count_with_multiplicities(L, self.m)
            self.comp_dics[n].dic+=[[L.copy(),p*self.p[cur_j]**high_n,count]]
            self.comp_dics[n].count+=count
            self.comp_dics[n].p+=count*p*self.p[cur_j]**high_n
            for j in range(cur_j, self.eta+1):
                L[j]='-'
        elif cur_j<self.eta:
            if p_limit//self.log_p[cur_j]>=high_n:
                for i in range(high_n+1):
                    L[cur_j]=i
                    self.create_par_comp_dic(n, p_limit-i*self.log_p[cur_j],cur_j+1,high_n-i,L,p*self.p[cur_j]**i, if_print=if_print)
    
    def par_comp_dic(self, n, c=0, if_ret=True, if_print=False):
        # creates a partial compact dictionary that contains all entries of the full compact dictionary which have
        # sampling probability of at least 2^(entropy*n+c).
        # If a partial compact dictionary already exists for a smaller c, the preexisting compact dictionary is used as
        # a base; 
        # otherwise, the old compact dictionary is overwritten
        if self.comp_dic_list(n).c<c:
            if self.eta==0:
                self.comp_dics[n]=par_comp_dic(c,[[n]])
            else:
                gc.collect()
                self.comp_dics[n]=par_comp_dic(-float('inf'),[],0,0)
                self.create_par_comp_dic(n,self.entropy*n+c,0,n,(self.eta+1)*['-'],1, if_print=if_print)
                self.comp_dics[n].dic.sort(key=lambda x: x[1], reverse=True)
                self.comp_dics[n].len=len(self.comp_dics[n].dic)
                self.comp_dics[n].c=c
                gc.collect()
                out=self.comp_dic_list(n)
        elif self.comp_dic_list(n).c>c:
            out_L=[]
            p=0
            count=0
            for l in self.comp_dic_list(n).dic:
                if largelog(l[1])<-self.entropy*n-c:
                    break
                out_L+=[l]
                p+=l[1]*l[2]
                count+=l[2]
            out=par_comp_dic(c, out_L, count, p)
        else:
            out=self.comp_dic_list(n)
        if if_ret:
            return out
        
    def comp_dic(self, n, if_ret=True, if_print=False):
        # creates the full compact dictionary
        c=-n*(self.entropy-max(self.log_p))+1
        return self.par_comp_dic(n, c=c, if_ret=if_ret, if_print=if_print)
    
    
    ### TABLE FUNCTIONS ###
    
    def raw_data(self, low_n, high_n, c=0, delete_after=False, step=1, aborts=True):
        # raw_data returns the success probability and complexities for all n in the range of [low_n, high_n]
        # in a csv-style table.
        
        # If aborts = True, the table is composed based on data from partial compact distionaries. This is faster,
        # but does not contain the expected runtime for the KeyGuess-Algorithm
        
        # If aborts = False, both expected data for KeyGuess and AbortedKeyGuess will be displayed, at the cost of
        # a significant runtime increase. Not recommended for wide distributions (like Falcon)
        if aborts:
            self.make_raw_data(low_n, high_n, c=c, delete_after=delete_after, step=step)
        else:
            self.make_raw_data_no_aborts(low_n, high_n, c=c, delete_after=delete_after, step=step)
    
    def make_raw_data(self, low_n, high_n, c=0, delete_after=False, step=1):
        print("n epsilon coresize Eclassic Equantum")
        for n in range(low_n, high_n+1, step):
            L=self.par_comp_dic(n, c=c, if_ret=True, if_print=False)
            p=0
            count=0
            ET=0
            Equantum=0
            for l in L.dic:
                p+=l[1]*l[2]
                ET+=l[1]*l[2]*(count+(l[2]+1)/2)
                Equantum+=l[1]*approx_sum_of_roots(count+1,count+l[2])
                count+=l[2]
            ET+=(1-p)*count
            print(n,format(float(p),'f'),largelog(count),largelog(ET),largelog(Equantum))
            if delete_after:
                del(self.comp_dics[n])
                gc.collect()
    
    def make_raw_data_no_aborts(self, low_n, high_n, c=0, delete_after=False, step=1):
        print("n epsilon coresize Eclassic Enoabort Equantum")
        for n in range(low_n, high_n+1, step):
            L=self.comp_dic(n, if_ret=True, if_print=False)
            p1=0
            p2=0
            count1=0
            count2=0
            ET1=0
            ET2=0
            Equantum=0
            for l in L.dic:
                if largelog(l[1])>=-self.entropy*n-c:
                    p1+=l[1]*l[2]
                    ET1+=l[1]*l[2]*(count1+(l[2]+1)/2)
                    Equantum+=l[1]*approx_sum_of_roots(count1+1,count1+l[2])
                    count1+=l[2]
#                else:
#                    ELZ1+=(1-p1)*l[2]
                p2+=l[1]*l[2]
                ET2+=l[1]*l[2]*(count2+(l[2]+1)/2)
                count2+=l[2]
            ET1+=(1-p1)*count1
            print(n,format(float(p1),'f'),largelog(count1),largelog(ET1),largelog(ET2),largelog(Equantum))
            if delete_after:
                del(self.comp_dics[n])
                gc.collect()
    
    
    ### OTHER FUNCTIONS ###
        
    def count_til_p(self,n,c=0,ret_l=True, if_print=True):
        # Returns the success probability of AbortedKeyGuess that occurs when AbortedKeyGuess is run
        # until the sampling probability goes below 2^(-H(.)n+c).
        if self.comp_dic_list(n).c<c:
            self.par_comp_dic(n,c,if_print=if_print)
            out=self.comp_dic_list(n).count
        elif self.comp_dic_list(n).c==c:
            out=self.comp_dic_list(n).count
        else:
            out=0
            i=0
            L=self.comp_dic_list(n).dic
            while i<len(L) and largelog(L[i][1])>-self.entropy*n-c:
                out+=L[i][2]
                i+=1
        if ret_l:
            return largelog(out)
        return out