1. Kaip atskirti parametru analize nuo klasteriu skaiciu pasirinkimo, jei skaiciaus pasirinkimui reikia istirti parametrus?
2. Ar galima naudoti k-medoids?
3. Ar galima naudoti kosinusu distancija? Jei ja naudosim k-medoids, ar reikia kituose (agnes) irgi ja naudoti? 
4. O minkovskio p = 0.5 distancija galima naudoti?
5. Isskirtys: ar galima salinti butent klasterizavimui, nes jos labai iskraipo klasterizavima? Ar palikti, bet tada k medoids ir agnes blogiai viekia. Ar reikia tada vietoj agnes kita algoritma panaudoti (SOM)? Ar tesiog naudoti cosine distance?

1. klasteriu skaiciu pasirinkti su default parametrais (euklidinis atstumas, single jungimas)
2. taip
3. ne. Euklidinis/manhatano 
4. minkovsio p=0.5 galima bet reikia ziureti (nenaudosim krc)
5. labai ekstremalias isskirtis pakeisti riba (3IQR?), jas zymeti ant projekciju

tikslas - gauti geriausia klasterizavimo rezultata ir pagal ji jau tikrinti

stabiluma tikrinam tik ant originalios aibes