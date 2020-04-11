"""
==================
Ejercicio Final
==================


"""

##
print(__doc__)

print('ECD2020')
print('Este ejercicio tiene dos maneras de resolverse.')
print('Lo tiene que tener listo para el fin de la Cuarentena')

print('Opción A: contestan todas las preguntas directamente en relación a cómo resolverían el problema.  Es decir explicando por qué lo que plantean podría funcionar.')
print('Opción B: elijan una (al menos) pregunta e intentan implementar una solución, codificando en R, Java o python.')

print('0 - Construyan una alternativa para detectar pestañeos y trabajen sobre el dataset de pestañeos para simular y testear el abordaje propuesto.')
print('1 - De las señales del EPOC Emotiv que obtuvimos de SUJETO, intenten estudiar las señales detectando: los pestañeos sobre F8 y F7, el momento donde el sujeto cierra los ojos, donde abre y cierra la boca, donde mueve la cabeza haciendo Roll, y donde mueve la cabeza haciendo YAW.')
print('2 - Sobre los datos de MNIST, intenten luego de clusterizar armar un clasificador.')
print('3 - Busquen un dataset de internet público de señales de sensores.  ¿Cómo lo abordarían exploratoriamente, qué procesamiento y qué análisis harían?')
print('4 - Prueben alternativas para mejorar la clasificación de las ondas alfa.')
print('5 - ¿Que feature utilizarian para mejorar la clasificacion que ofrece Keras con MLP para las series de tiempo?')

# El experimento que hicimos con SUJETO está en el directorio data/experimentosujeto.dat

'''
El formato de los datos es

        "COUNTER",
        "AF3",
        "F7",
        "F3",
        "FC5",
        "T7",
        "P7",
        "O1",
        "O2",
        "P8",
        "T8",
        "FC6",
        "F4",
        "F8",
        "AF4",
        "GYRO_X",
        "GYRO_Y",
        "RESERVED",
        "RESERVED",
        "RESERVED",
        "RESERVED",
        "RESERVED"

Los datos buenos que tomamos deberían ser F7 y F8, GYRO_X y GYRO_Y.

'''



import matplotlib.pyplot as plt
import numpy as np
# In[1]:
import pandas as pd

## %%

signals = pd.read_csv('data/experimentosujeto.dat', delimiter=' ', names = [
            "COUNTER",
        "AF3",
        "F7",
        "F3",
        "FC5",
        "T7",
        "P7",
        "O1",
        "O2",
        "P8",
        "T8",
        "FC6",
        "F4",
        "F8",
        "AF4",
        "GYRO_X",
        "GYRO_Y",
        "RESERVED1",
        "RESERVED2",
        "RESERVED3",
        "RESERVED4",
        "RESERVED5"])

## %%
signals.shape

# In[1]:
data = signals.values

eeg = data[:,8]

plt.plot(eeg,'r', label='EEG')
plt.xlabel('t');
plt.ylabel('eeg(t)');
plt.title(r'EEG Signal')     # r'' representa un raw string que no tiene caracteres especiales
plt.ylim([-2000, 2000]);
plt.xlim([0,len(eeg)])
plt.show()


## %%
data.shape

## %%
import mne

data = signals.values

ch_names = [
            "COUNTER",
        "AF3",
        "F7",
        "F3",
        "FC5",
        "T7",
        "P7",
        "O1",
        "O2",
        "P8",
        "T8",
        "FC6",
        "F4",
        "F8",
        "AF4",
        "GYRO_X",
        "GYRO_Y",
        "RESERVED1",
        "RESERVED2",
        "RESERVED3",
        "RESERVED4",
        "RESERVED5"]

sfreq = 128
data =data[:,list([2,7,8,13,15,16])]

ch_renames = [ch_names[2],ch_names[7],ch_names[8],ch_names[13],ch_names[15],ch_names[16]]
ch_types = ['eeg' for _ in ch_renames]



info = mne.create_info(ch_renames, sfreq, ch_types=ch_types)

raw = mne.io.RawArray(data.T, info)
#raw.add_events(events)

raw.plot_psd()

raw.filter(1,20)

raw.plot_psd()


## %%
raw.plot(scalings='auto', block=True)

## %%
#Intentaremos resolver este problema, como primer paso.
print(r'Resolucion: "0 - Construyan una alternativa para detectar pestañeos y trabajen sobre el dataset de pestañeos para simular y testear el abordaje propuesto."')

signals.shape
## %%

#Ploteamos todos los valores, sin usar mne ni ninguna otra libreria especial.
plt.plot(signals.O1,'g',label='EEG O1')
plt.plot(signals.O2,'g',label='EEG O2')
plt.plot(signals.F7,'b',label='EEG F7')
plt.plot(signals.F8,'b',label='EEG F8')
plt.plot(signals.GYRO_X,'r',label='Giro X')
plt.plot(signals.GYRO_Y,'r',label='Giro Y')

plt.show()

#De esto, descubrimos que cada señal esta en su propio nivel de Hz. Entonces, antes de nada, lo que deberiamos hacer es lo siguiente:
# 1 - Normalizar cada serie de tiempo, verificando que no faltan valores
# 2 - Poner filtros, de ser necesario.
# 3 - Identificar que valores estan como outliers en cada onda

## %%
# Hacemos: 1 - Normalizar cada serie de tiempo, verificando que no faltan valores

def identifyFaultySignal(counterColumn,freqIn):
        freq = freqIn + 1  # porque es el punto de "quiebre de la busqueda" de continuidad, no la frecuencia real de entrada.
        count_missing = 0
        last = counterColumn[0]

        for row in counterColumn:

                if (row < last):
                        count = freq + row - last
                else:
                        count = row - last

                count_missing += count

                last = row + 1
                if (last == freq):
                        last = 0

        return count_missing

print(" La medicion tiene los siguientes saltos en la señal: %2d" % identifyFaultySignal(signals.COUNTER, 128)) #<-Esto nos da que no hay ningun salto.

## %%
# El ejercicio anterior, nos dio que no hay saltos (verificar corriendolo) en la señal. Por lo tanto, progresamos hacia el paso 2.
# Hacemos: 2 - Poner filtros, de ser necesario. <- para nuestro primer analisis, intentaremos estudiar la onda "as-is"
# Hacemos: 3 - Identificar que valores estan como outliers en cada onda <- para esto, primero, identificamos los elementros representativos de cada sample estadistico.

quartils = [0.25,0.5,0.75,0.99] # <- luego de avanzar en el TP, me doy cuenta que estamos buscando outliers; en una señal neurologica como esta deberian ser casos mucho mas raros. El cuarto cuartil deberia estar en mas de 90.

O1 = signals.O1
O2 = signals.O2

F7 = signals.F7
F8 = signals.F8

Gx = signals.GYRO_X
Gy = signals.GYRO_Y

qrt_O1 = np.quantile(O1,quartils)
qrt_O2 = np.quantile(O2,quartils)
qrt_F7 = np.quantile(F7,quartils)
qrt_F8 = np.quantile(F8,quartils)
qrt_Gx = np.quantile(Gx,quartils)
qrt_Gy = np.quantile(Gy,quartils)

## %%

plt.boxplot([O1, O2], showmeans=True, whis=99)
plt.xticks([1,2],['O1','O2'])

plt.show()

## %%
plt.close()

plt.boxplot([F7,F8], showmeans=True, whis=99)
plt.xticks([1,2],['F7','F8'])

plt.show()

## %%
plt.close()

plt.boxplot([Gx,Gy], showmeans=True, whis=99)
plt.xticks([1,2],['Gyro X', 'Gyro Y'])

plt.show()

## %%
plt.close()

#Lo que se observa en todos los casos, es que la señal es, normalmente, muy regular. Con lo cual, con identificar los picos, deberia bastarnos.

## %%
# Ahora, aplicaremos los filtros a cada tipo de señal; entendemos que vamos a necesitaremos filtros espectrales para esto.
# Habiendo revisado "contadoreventos.py", concluimos:

print ('Total observaciones: %d' % len(O1))
print ('Cuartil superior: %d' % qrt_O1[3])

peaks_O1 = O1[ O1 > qrt_O1[3]] #Los picos de valores

print ('Picos: %d' % len(peaks_O1))

deriv_O1 = np.diff(O1)

print ('Derivadas: %d' % len(deriv_O1))

positivos_deriv_O1 = np.where(deriv_O1 == 0)
positivos_deriv_O1_ind = positivos_deriv_O1[0]

print ('Derivadas positivas: %d' % len(positivos_deriv_O1_ind))

finalresult_O1 = np.isin(positivos_deriv_O1_ind, peaks_O1.index)

blinkings = finalresult_O1.sum()
posiciones = positivos_deriv_O1_ind[finalresult_O1] # para que me devuelva las posiciones en el grafico.

print ('Blinkings: %d' % blinkings)
print ('Locations:');print(posiciones)

## %%
#Ahora, armamos una funcion que nos devuelva las posiciones con los picos. De esta manera, podremos reutilizarla con todas las demas mediciones.
def returnPeaks(signal, quartile):

        peaks = signal[signal > quartile[3]]  # Los picos de valores
        deriv = np.diff(signal)

        positivos_deriv = np.where(deriv == 0)
        positivos_deriv_ind = positivos_deriv[0]

        finalresult = np.isin(positivos_deriv_ind, peaks.index)

        posiciones = positivos_deriv_ind[finalresult]  # para que me devuelva las posiciones en el grafico.

        return posiciones

## %%
#O1 y O2 son occipitales, por lo tanto te pueden servir para identificar cuando una persona tiene los ojos cerrados. (pestañeos)
#Eso lo haces calculando el feature PSD sobre cada uno de los dos.
from signalfeatureclassification import psd

#Con el PSD podemos identificar un ratio de comparacion de densidad de la señal. (estan todas en 128hz, segun entendi)
psd_O1 = psd(O1[0:500])
psd_O1_2 = psd(O1[501:1000])
#Hemos identificado que necesitamos esto para poder comparar las diferentes curvas en rangos de lecturas. Muy interesante!

## %% we use this functions to find all the divisors of the given reading [1, 2, 23, 46, 419, 838, 9637, 19274], we decide to take buckets of 46 observations.
def allDivisors(number):
    divisors = []
    for i in range(1, number + 1):
        if number % i == 0:
            divisors.append(i)

    return divisors

divisors = allDivisors(19274)
divisors

## %%
# Now we got the bucket for observations of 46 to analize; let's produce arrays of psds for all the sensors.
def get_all_psd_by_window(signals, window_size):

    elements_count = signals.size
    hops = range(0,elements_count, window_size)
    result_psd = []

    for initial in hops:
        final = initial + window_size
        result_psd.append(psd(signals[initial:final]))

    return result_psd

## %%
window = 46

psd_O1 = np.asarray(get_all_psd_by_window(O1, window))
psd_O2 = np.asarray(get_all_psd_by_window(O2, window))
psd_F7 = np.asarray(get_all_psd_by_window(F7, window))
psd_F8 = np.asarray(get_all_psd_by_window(F8, window))
psd_Gx = np.asarray(get_all_psd_by_window(Gx, window))
psd_Gy = np.asarray(get_all_psd_by_window(Gy, window))

## %%
# Ahora, vamos a armar listas de indices con picos por cada uno de los sensores.

qrt_psd_O1 = np.quantile(psd_O1,quartils)
qrt_psd_O2 = np.quantile(psd_O2,quartils)
qrt_psd_F7 = np.quantile(psd_F7,quartils)
qrt_psd_F8 = np.quantile(psd_F8,quartils)
qrt_psd_Gx = np.quantile(psd_Gx,quartils)
qrt_psd_Gy = np.quantile(psd_Gy,quartils)

## %% this defines a specific peaks identification function for PSDs kind of ndarrays
def returnPeaksPSD(sig, quartile):
    signal = pd.Series(sig)
    peaks = signal[signal > quartile[3]]  # Los picos de valores
    return peaks

## %% identificamos todos los picos mediante PSD.

peaks_O1 = returnPeaksPSD(psd_O1, qrt_psd_O1)
peaks_O2 = returnPeaksPSD(psd_O2, qrt_psd_O2)
peaks_F7 = returnPeaksPSD(psd_F7, qrt_psd_F7)
peaks_F8 = returnPeaksPSD(psd_F8, qrt_psd_F8)
peaks_Gx = returnPeaksPSD(psd_Gx, qrt_psd_Gx)
peaks_Gy = returnPeaksPSD(psd_Gy, qrt_psd_Gy)

## %%
#Ahora analizamos si tenemos picos en los mismos sensores en el mismo momento en el que se da en la serie de tiempo.

# "F7 y F8 están ubicados en la zona frontal, por lo que van a detectar mejor los picos provocados por los pestañeos."
#Hacemos, entonces, el mismo analisis sobre F7 y F8.

maximum_values_Fn = np.isin(peaks_F7.index, peaks_F8.index)

#El punto en el que la persona pestaneo deberia ser este en la matriz PSD para F7 y F8
peaks_F7[maximum_values_Fn].index

#Los rangos de puntos deberian ser:

initial_Fn = peaks_F8[maximum_values_Fn].index * window # lo multiplico por la ventana para que me retorne la posicion original en la serie de tiempo.

print("Los pestañeos se dieron, en la serie de tiempo en: ")

for i in initial_Fn:
    print("El pestañeo COMIENZA en %d y termina en %d en la serie de tiempo original." % (i, i+window))

## %% Intentamos resolver - Ejercicio 1 - De las señales del EPOC Emotiv que obtuvimos de SUJETO, intenten estudiar las señales detectando: los pestañeos sobre F8 y F7, el momento donde el sujeto cierra los ojos, donde abre y cierra la boca, donde mueve la cabeza haciendo Roll, y donde mueve la cabeza haciendo YAW."
# # En el punto anterior, resulvimos con F7 y F8 los pestañeos. Veamos ahora si podemos resolver cuando cierra y abre los ojos (Sensores O1 y O2)
# "Cuando ese valor [el PSD en O1 y O2] sea máximo podés asumir que es cuando la persona tiene los ojos cerrados."

maximum_values_On = np.isin(peaks_O1.index, peaks_O2.index)

#El punto en el que la persona movio el parpado deberia ser este en la matriz PSD para O1 y O2
peaks_O1[maximum_values_On].index

#Los rangos de puntos deberian ser:

initial_On = peaks_O1[maximum_values_On].index * window


## %% Resultado final - Identificacion de apertura y cierre de ojos.
print("Los ojo se abren/cierran  en la serie de tiempo en: ")

if ((initial_On.size % 2) == 0):
    print ("Podemos asumir que la persona empezo abriendo los ojos y termino cerrandolos o alreves")
else:
    print("O la persona esta con los ojos abiertos, o con los ojos cerrados")

for i in initial_On:
    print(" El 'parpado se mueve' COMIENZA en %d y termina en %d" % (i, i+window))

## %% vamos a intentar identificar cuando mueve la cabeza. Para esto, deberiamos ver el giroscopio X e Y.
if (peaks_Gx.size < peaks_Gy.size):
    excluir = np.isin(peaks_Gx.index, peaks_Gy.index) # cuando se giran ambos ejes - deberiamos excluir estos casos.
else:
    excluir = np.isin(peaks_Gy.index, peaks_Gx.index)  # cuando se giran ambos ejes - deberiamos excluir estos casos.

# nota - nos dio all false, pero vamos a implementar igual la logica de exclusion de esos casos:
maximum_values_Gn_yaw = peaks_Gx[excluir != True].index #cuando se dan maximos movimientos en X - YAW
maximum_values_Gn_roll = peaks_Gy[excluir != True].index #cuando se dan maximos movimientos en Y - ROLL


## %%
print("Los momentos en los que se hacen acciones con la cabeza: ")

yaws = peaks_Gx[maximum_values_Gn_yaw].index * window
rolls = peaks_Gy[maximum_values_Gn_roll].index * window

print("YAWS: ")
for i in yaws:
    print("COMIENZA en %d y termina en %d" % (i, i+window))

print("ROLLS: ")
for i in rolls:
    print("COMIENZA en %d y termina en %d" % (i, i+window))

# NOTA: No entendemos como detectar si abrio la boca o no.

## %% Ejercicio 2 - Sobre los datos de MNIST, intenten luego de clusterizar armar un clasificador.





