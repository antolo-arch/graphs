import networkx as nx
import random
import numpy as np
import math
from timeit import default_timer as timer
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline


random.seed(5)
def generowanie_tabeli(n):
    #obliczanie ilosci krawedzi
    ilosc_krawedzi = math.ceil(n*(n-1)/4)
    #tworzenie tablicy n na n wypelnionej zerami
    tabela = np.zeros((n,n), dtype = "int")
    krawedzie = 0
    for i in range(1,n-1):
        tabela[random.randint(0,i-1),i] = 1
        krawedzie+=1
    while krawedzie<ilosc_krawedzi:
        #losowe wybieranie punktu w gornym trojkacie macierzy
        x = random.randint(0,n-2)
        y = random.randint(x+1,n-1)
        if tabela[x,y] ==0:
            krawedzie+=1
            tabela[x,y]=1
    return tabela
 
class graph:
 
    def __init__(self, macierz_sasiedztwa = None):
        self.vertexes = {} #lista nastepnikow jako slownik
        self.macierz_sasiedztwa = macierz_sasiedztwa;
        #jesli macierz sasiedztwa jest na poczatku podana dodajemy wszystkie krawedzie
        if self.macierz_sasiedztwa is not None:
            for numer_rzedu, rzad in enumerate(self.macierz_sasiedztwa):
                for numer_kolumny, istnieje in enumerate(rzad):
                    if istnieje == 1:
                        self.insert_edge((numer_rzedu,numer_kolumny))
        self.edges = self.tabela_krawedzi() 
        self.wierzcholki = list(self.vertexes.keys())
 
 
    def insert_edge(self, edge):
        vertex_1, vertex_2 = edge
        if vertex_1 in self.vertexes:
            #wstawianie wartosci do slownika, jezeli dany wierzcholek wczesniej istnial dodajemy nowy wierzcholek do jego wartosci
            self.vertexes[vertex_1].append(vertex_2)
        else:
            #jezeli nie istnial inicjalizujemy nowy z wierzcholkiem w tablicy
            self.vertexes[vertex_1] = [vertex_2]
        if  vertex_2 not in self.vertexes:
            #jezeli drugi wierzcholek nie istnial w liscie, inicjalizujemy go z pusta tablica
            self.vertexes[vertex_2] = []
 
    #wyswietlanie slownika
    def lista_nastepnikow(self):
        for i in self.vertexes:
            print(i, ":", self.vertexes[i])
 
    #tworzenie tabeli krawedzi z listy nastepnikow
    def tabela_krawedzi(self):
        krawedzie = []
        for i,krawedz_1 in enumerate(self.vertexes):
            for krawedz_2 in list(self.vertexes.values())[i]:
                krawedzie.append((krawedz_1,krawedz_2))
        return krawedzie
 
    def bfs_nastepniki(self, start = None):
        if start is None:
            start = self.najwiecej_wyjsc()
            #sprawdzamy, z którego wierzchołka wychodzi najwięcej krawędzi, od niego zaczynamy
        visited = [start]
        stack = []
        droga = [start]
        nieodwiedzone = list(self.wierzcholki)
        nieodwiedzone.remove(start)
        #do stosu dodajemy nastepne wierzcholki i oznaczamy je jako odwiedzone
        for i in self.vertexes[start]:
            if i not in visited:
                stack.append(i)
                visited.append(i)
                nieodwiedzone.remove(i)
        #dopoki stos nie jest pusty sprawdzamy czy wierzcholek ma nastepnikow i czy nie byly one wczesniej odwiedzane
        #wybieramy 1 wierzcholek ze stosu, byl on dodany jako pierwszy przez poprzednikow
        while (len(stack)>0):
            droga.append(stack[0])
            for i in self.vertexes[stack[0]]:
                if i not in visited:
                    stack.append(i)
                    visited.append(i)
                    nieodwiedzone.remove(i)
            stack.pop(0)
            if(len(stack)==0 and len(nieodwiedzone)!=0):
                stack.append(nieodwiedzone[0])
                nieodwiedzone.pop(0)
            #wierzcholek wykorzystany usuwamy
        return droga
    
    def bfs_krawedzie(self,start = None):
        if start is None:
            start = self.najwiecej_wyjsc()
        visited = [start]
        stack = []
        droga = [start]
        nieodwiedzone = list(self.wierzcholki)
        nieodwiedzone.remove(start)
        for v1,v2 in self.edges:
            if v1 == start:
                stack.append(v2)
                visited.append(v2)
                nieodwiedzone.remove(v2)
        while (len(stack)>0):
            droga.append(stack[0])
            tmp = stack[0]
            for v1,v2 in self.edges:
                #jezeli 1 krawedz jest taka sama jak nasza zmienna pomocnicza a krawedz druga nie jest odwiedzona
                #to dodajemy ja na stos
                if v1 == tmp and v2 not in visited:
                    stack.append(v2)
                    visited.append(v2)
                    nieodwiedzone.remove(v2)
            stack.pop(0)
            if(len(stack)==0 and len(nieodwiedzone)!=0):
                stack.append(nieodwiedzone[0])
                nieodwiedzone.pop(0)
                
        return droga
        
        
    def bfs_macierz(self,start = None):
        if start is None:
            start = self.najwiecej_wyjsc()
        visited = [start]
        stack = []
        droga = [start]
        nieodwiedzone=list(self.wierzcholki)
        nieodwiedzone.remove(start)
        for i,v2 in enumerate(self.macierz_sasiedztwa[start]):
            if v2 == 1:
                stack.append(i)
                visited.append(i)
                nieodwiedzone.remove(i)
        while(len(stack)>0):
            droga.append(stack[0])
            v1 = stack[0]
            for i,v2 in enumerate(self.macierz_sasiedztwa[v1]):
                #jezeli w macierzy krawedz istnieje i nie zostala odwiedzona to dodajemy ja na stos
                if v2 == 1 and i not in visited:
                    stack.append(i)
                    visited.append(i)
                    nieodwiedzone.remove(i)
            stack.pop(0)
            if(len(stack)==0 and len(nieodwiedzone)!=0):
                stack.append(nieodwiedzone[0])
                nieodwiedzone.pop(0)
        return droga
            
        
    def dfs_nastepniki(self, start = None):
        if start is None:
            start = self.najwiecej_wyjsc()
        visited = [start]
        stack = []
        droga = [start]
        nieodwiedzone=list(self.wierzcholki)
        nieodwiedzone.remove(start)
        for i in self.vertexes[start]:
            if i not in visited:
                stack.append(i)
                visited.append(i)
                nieodwiedzone.remove(i)
        while (len(stack)>0):
            droga.append(stack[0])
            vertex = stack[0]
            stack.pop(0)
            # w tym wypadku trzeba usuwac przed wykonaniem funkcji, ponieważ dodajemy kolejne wierzchołki na 1 pozycję
            #by najpierw isc w głąb grafu
            for i in self.vertexes[vertex]:
                if i not in visited:
                    stack.insert(0,i)
                    visited.append(i)
                    nieodwiedzone.remove(i)
            if(len(stack)==0 and len(nieodwiedzone)!=0):
                stack.append(nieodwiedzone[0])
                nieodwiedzone.pop(0)
        return droga
    
    def dfs_krawedzie(self,start = None):
        if start is None:
            start = self.najwiecej_wyjsc()
        visited = [start]
        stack = []
        droga = [start]
        nieodwiedzone=list(self.wierzcholki)
        nieodwiedzone.remove(start)
        for v1,v2 in self.edges:
            if v1 == start:
                stack.append(v2)
                visited.append(v2)
                nieodwiedzone.remove(v2)
        while (len(stack)>0):
            droga.append(stack[0])
            tmp = stack[0]
            stack.pop(0)
            for v1,v2 in self.edges:
                if v1 == tmp and v2 not in visited:
                    stack.insert(0,v2)
                    visited.append(v2)   
                    nieodwiedzone.remove(v2)
            if(len(stack)==0 and len(nieodwiedzone)!=0):
                stack.append(nieodwiedzone[0])
                nieodwiedzone.pop(0)
        return droga
        
        
    def dfs_macierz(self,start = None):
        if start is None:
            start = self.najwiecej_wyjsc()
        visited = [start]
        stack = []
        droga = [start]
        nieodwiedzone=list(self.wierzcholki)
        nieodwiedzone.remove(start)
        for i,v2 in enumerate(self.macierz_sasiedztwa[start]):
            if v2 == 1:
                stack.append(i)
                visited.append(i)
                nieodwiedzone.remove(i)
        while(len(stack)>0):
            droga.append(stack[0])
            v1 = stack[0]
            stack.pop(0)
            for i,v2 in enumerate(self.macierz_sasiedztwa[v1]):
                if v2 == 1 and i not in visited:
                    stack.insert(0,i)
                    visited.append(i)
                    nieodwiedzone.remove(i)
            if(len(stack)==0 and len(nieodwiedzone)!=0):
                stack.append(nieodwiedzone[0])
                nieodwiedzone.pop(0)
        return droga
 
    def sortowanie_1_lista(self):
        odwiedzone  = []
        nieodwiedzone= list(self.vertexes.keys())
        #tworzymy liste wszystkich wierzcholkow
        wynik = []
        #dopoki sa jakies nieodwiedzone wierzcholki wykonujemy funkcje sort top
        while(len(nieodwiedzone)>0):
            for i in nieodwiedzone:
                self.sort_top_lista(i,wynik,odwiedzone,nieodwiedzone)
                if "circle" in wynik:
                    print("istnieje obwod")
                    return None
        return wynik
 
    def sortowanie_1_krawedzie(self):
        odwiedzone = []
        nieodwiedzone = []
        for x,y in self.edges:
            if x not in nieodwiedzone:
                nieodwiedzone.append(x)
            if y not in nieodwiedzone:
                nieodwiedzone.append(y) 
        wynik = []
        while(len(nieodwiedzone)>0):
            for i in nieodwiedzone:
                self.sort_top_krawedzie(i,wynik,odwiedzone,nieodwiedzone)
                if "circle" in wynik:
                    print("istnieje obwod")
                    return None
        return wynik
    def sortowanie_1_macierz(self):
        odwiedzone = []
        nieodwiedzone = []
        wynik=[]
        for i in range(len(our_graph.macierz_sasiedztwa)):
            nieodwiedzone.append(i)
        while(len(nieodwiedzone)>0):
            for i in nieodwiedzone:
                self.sort_top_macierz(i,wynik,odwiedzone,nieodwiedzone)
                if "circle" in wynik:
                    print("istnieje obwod")
                    return None
        return wynik
    def sort_top_lista(self,v,wynik,odwiedzone,nieodwiedzone):
 
        #ustalamy czy wierzzcholek byl wczesniej odwiedzony i usuwamy go z nieodwiedzonych
        odwiedzone.append(v)
        nieodwiedzone.remove(v)
 
        #przszukujemy nastepnikow wierzcholka v i funkcje powtarzamy rekurencyjnie dla kazdego nastepnika
        for u in self.vertexes[v]: 
            if u in odwiedzone and u not in wynik:
                wynik.append("circle")
                break                
            if u in nieodwiedzone:
                self.sort_top_lista(u,wynik,odwiedzone,nieodwiedzone)
        #na poczatku dodamy wierzcholek, ktory zostal odwiedzony jako ostatni, czyli bedzie on na koncu
        #bo nie ma zadnego nastepnika
        wynik.insert(0,v)
 
    def sort_top_krawedzie(self,v,wynik,odwiedzone,nieodwiedzone):
 
        odwiedzone.append(v)
        nieodwiedzone.remove(v)
 
        for x,y in self.edges: 
            if x ==v and y in odwiedzone and y not in wynik:
                wynik.append("circle")
                break                
            if x ==v and y in nieodwiedzone:
                self.sort_top_lista(y,wynik,odwiedzone,nieodwiedzone)
 
        wynik.insert(0,v)
    def sort_top_macierz(self,v,wynik,odwiedzone,nieodwiedzone):
 
        odwiedzone.append(v)
        nieodwiedzone.remove(v)
 
        for i in range(len(our_graph.macierz_sasiedztwa[v])):
            if self.macierz_sasiedztwa[v,i] == 1 and i in odwiedzone and i not in wynik:
                wynik.append("circle")
                break                
            if self.macierz_sasiedztwa[v,i] == 1 and i in nieodwiedzone:
                self.sort_top_macierz(i,wynik,odwiedzone,nieodwiedzone)
 
        wynik.insert(0,v)
 
 
 
    def sortowanie_2_lista(self):
        stopnie_in_wierzcholkow = {}
        wchodzace_wierzcholki = [item for sublist in list(self.vertexes.values()) for item in sublist]
        #linijka wyzej jest rownowazna temu:
        #for sublist in list(self.vertexes.values()):
        #   for item in sublist:
        #       wchodzace_wierzcholki.append(item)
        #na podstawie tabeli wchodzace wierzcholki liczymy ile razy dana krawedz pojawila sie jako "cel"
        #podrozy i na tej podstawie okreslamy jej stopien wejscia(ile krawedzi do niej wchodzi)
        for vertex in list(self.vertexes.keys()):
            stopnie_in_wierzcholkow[vertex] = wchodzace_wierzcholki.count(vertex)
        rozwiazanie = []
        #dzialamy dopoki rozwiazanie bedzie zawieralo wszystkie wierzcholki lub nie bedzie wierzcholka o stopniu
        #0  i nie bedzie mozna zmniejszyc pozostalych, a wiec bedzie petla
        #gdy rozwiazanie bedzie krotsze niz ilosc wierzcholkow wiemy ze byla tam petla
        while(len(rozwiazanie)!=len(stopnie_in_wierzcholkow) and 0 in stopnie_in_wierzcholkow.values()):
            for vertex in stopnie_in_wierzcholkow:
                if (stopnie_in_wierzcholkow[vertex] == 0):
                    rozwiazanie.append(vertex)
            #jezeli do wierzcholka nic nie wchodzi dodajemy go do rozwiazania, a od stopnia jego nastepnikow odejmujemy 1
                    for pom in self.vertexes[vertex]:
                        stopnie_in_wierzcholkow[pom]-=1
 
                    stopnie_in_wierzcholkow[vertex]=-1
            #ustawiamy wartosc wierzcholka w rozwiazaniu na -1 by bylo wiadomo ze zostal on juz w pelni wykorzystany
        return(rozwiazanie)
 
    def sortowanie_2_krawedzie(self):
 
        stopnie_in_wierzcholkow = {}
        wchodzace_wierzcholki = [x[1] for x in our_graph.edges]
        nieodwiedzone = []
        for x,y in self.edges:
            if x not in nieodwiedzone:
                nieodwiedzone.append(x)
            if y not in nieodwiedzone:
                nieodwiedzone.append(y) 
        for vertex in range(len(nieodwiedzone)):
            stopnie_in_wierzcholkow[vertex] = wchodzace_wierzcholki.count(vertex)
        rozwiazanie = []
        while(len(rozwiazanie)!=len(stopnie_in_wierzcholkow) and 0 in stopnie_in_wierzcholkow.values()):
            for vertex in stopnie_in_wierzcholkow:
                if (stopnie_in_wierzcholkow[vertex] == 0):
                    rozwiazanie.append(vertex)
                    for x,y in self.edges:
                        if x ==vertex:
                            stopnie_in_wierzcholkow[y]-=1
 
                    stopnie_in_wierzcholkow[vertex]=-1
        return(rozwiazanie)
 
    def sortowanie_2_macierz(self):
        stopnie_in_wierzcholkow = {}
        wchodzace_wierzcholki = []
        for i in self.macierz_sasiedztwa:
            for x in range(len(i)):
                if i[x] ==1:
                    wchodzace_wierzcholki.append(x)
        for i in range(len(self.macierz_sasiedztwa)):
            stopnie_in_wierzcholkow[i]=wchodzace_wierzcholki.count(i)
        rozwiazanie = []
        while(len(rozwiazanie)!=len(stopnie_in_wierzcholkow) and 0 in stopnie_in_wierzcholkow.values()):
            for vertex in stopnie_in_wierzcholkow:
                if (stopnie_in_wierzcholkow[vertex] == 0):
                    rozwiazanie.append(vertex)
                    for y in range(len(self.macierz_sasiedztwa[vertex])):
                        if self.macierz_sasiedztwa[vertex][y] ==1:
                            stopnie_in_wierzcholkow[y]-=1
 
                    stopnie_in_wierzcholkow[vertex]=-1
        return(rozwiazanie)
 
    def najwiecej_wyjsc(self):
        maks = 0
        for i, x in enumerate(list(self.vertexes.values())):
            if(len(x)>maks):
                maks = len(x)
                wynik = list(self.vertexes.keys())[i]
        return wynik
                    
while True:
    flag = 0
    flaga_macierz = 0
    print("Jaki graf chcesz tworzyć? (wybierz cyfrę)\n\t1) Wpisany z klawiatury,\n\t2) losowy.\nAby zakończyć działanie programu, wpisz 'exit'.\nAby zresetowac menu, wpisz 'home'")
    opcja = input()
    if opcja == 'exit':
        break
    elif opcja == 'home':
        continue
    elif opcja == '1':
        print("\nZ ilu wierzchołków chcesz utworzyć graf? (wpisz cyfrę)")
        pom = int(input())
        adj_matrix = []
        print("\nWprowadź z klawiatury macierz sąsiedztwa")
        for i in range(pom):
            flaga_macierz = 1
            x = list(map(int,input().split()))
            adj_matrix.append(x)
        our_graph = graph(adj_matrix)
    elif opcja == '2':
        print("\nZ ilu wierzchołków ma składać się losowo utworzony graf? (wpisz cyfrę)")
        pom = int(input())
        our_graph = graph(generowanie_tabeli(pom))
    else:
        print("\nWybrano nieprawidłową operację!")
        continue
    while True:
        print("\nJaką operację na grafie chcesz wykonać? (wybierz cyfrę)\n\t1) Narysuj graf (tylko dla małej liczby wierzchołków),\n\t2) wyświetl listę następników,\n\t3) wyświetl tabelę krawędzi,\n\t4) wyświetl macierz sąsiedztwa,\n\t5) BFS,\n\t6) DFS,\n\t7) posortuj graf topologicznie przeszukując 'w głąb',\n\t8) posortuj graf topologicznie przeszkując 'wszerz',\nAby zakończyć działanie programu, wpisz 'exit'.\nAby zresetowac menu, wpisz 'home'")
        operacja = input()
        if operacja == 'exit':
            flag = 1
            break
        elif operacja == 'home':
            flag = 2
            break
        elif operacja == '1':
            G1 = nx.DiGraph()
            nodes = our_graph.vertexes.keys()
            G1.add_nodes_from(nodes)  
            G1.add_edges_from(our_graph.edges)
            nx.draw(G1,pos=nx.circular_layout(G1),with_labels=True)
            plt.show()
        elif operacja == '2':
            print("\nLista następników:")
            our_graph.lista_nastepnikow()
        elif operacja == '3':
            print("\nTabela krawędzi:")
            tabela = our_graph.tabela_krawedzi()
            for i in tabela:
                print(i)
        elif operacja == '4':
            print("\nMacierz sąsiedztwa:")
            if flaga_macierz == 1:
                for i in adj_matrix:
                    print(*i)
            else:
                print(our_graph.macierz_sasiedztwa)
        elif operacja == '5':
            print("\nBFS:")
            droga = our_graph.bfs_nastepniki()
            print(droga)
        elif operacja == '6':
            print("\nDFS:")
            droga = our_graph.dfs_nastepniki()
            print(droga)
        elif operacja == '7':
            print("\nSortowanie topoogiczne przeszukując 'w głąb':")
            res = our_graph.sortowanie_1_lista()
            print(res)
        elif operacja == '8':
            print("\nSortowanie topoogiczne przeszukując 'wszerz':")
            res = our_graph.sortowanie_2_lista()
            print(res)
    if flag == 1:
        break
    if flag == 2:
        continue

    
'''x = np.array([100,366,633,900,1166,1433,1700,1966,2233,2500])
wyniki_sortowanie_1_lista = []
wyniki_sortowanie_1_krawedzie = []
wyniki_sortowanie_1_macierz = []
wyniki_sortowanie_2_lista = []
wyniki_sortowanie_2_krawedzie = []
wyniki_sortowanie_2_macierz = []

for i in x:
    our_graph = graph(generowanie_tabeli(i))
    
    start = timer()
    our_graph.sortowanie_1_lista()
    end = timer()
    wyniki_sortowanie_1_lista.append(end-start)
    
    start = timer()
    our_graph.sortowanie_1_krawedzie()
    end = timer()
    wyniki_sortowanie_1_krawedzie.append(end-start)
    
    start = timer()
    our_graph.sortowanie_1_macierz()
    end = timer()
    wyniki_sortowanie_1_macierz.append(end-start)
    
    start = timer()
    our_graph.sortowanie_2_lista()
    end = timer()
    wyniki_sortowanie_2_lista.append(end-start)
    
    start = timer()
    our_graph.sortowanie_2_krawedzie()
    end = timer()
    wyniki_sortowanie_2_krawedzie.append(end-start)
    
    start = timer()
    our_graph.sortowanie_2_macierz()
    end = timer()
    wyniki_sortowanie_2_macierz.append(end-start)
    

    
xnew = np.linspace(x.min(),x.max(),5000)
     
plt.figure()
plt.title("sortowanie topologiczne 'w głąb'")
plt.xlabel("ilość wierzchołków")
plt.ylabel("czas w sekundach")

spl = make_interp_spline(x, np.array(wyniki_sortowanie_1_lista), k=3)
y_smooth = spl(xnew)
plt.plot(xnew, y_smooth)

spl = make_interp_spline(x, np.array(wyniki_sortowanie_1_krawedzie), k=3)
y_smooth = spl(xnew)
plt.plot(xnew, y_smooth)

spl = make_interp_spline(x, np.array(wyniki_sortowanie_1_macierz), k=3)
y_smooth = spl(xnew)
plt.plot(xnew, y_smooth)

plt.legend(["lista następników","tabela krawędzi","macierz sąsiedztwa"],loc='upper left')
plt.savefig("sortowanie_w_glab.pdf")

####################################################

plt.figure()
plt.title("sortowanie topologiczne 'wszerz'")
plt.xlabel("ilość wierzchołków")
plt.ylabel("czas w sekundach")

spl = make_interp_spline(x, np.array(wyniki_sortowanie_2_lista), k=3)
y_smooth = spl(xnew)
plt.plot(xnew, y_smooth)

spl = make_interp_spline(x, np.array(wyniki_sortowanie_2_krawedzie), k=3)
y_smooth = spl(xnew)
plt.plot(xnew, y_smooth)

spl = make_interp_spline(x, np.array(wyniki_sortowanie_2_macierz), k=3)
y_smooth = spl(xnew)
plt.plot(xnew, y_smooth)

plt.legend(["lista następników","tabela krawędzi","macierz sąsiedztwa"],loc='upper left')
plt.savefig("sortowanie_wszerz.pdf")
'''