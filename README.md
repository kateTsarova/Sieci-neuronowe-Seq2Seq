# Sieci Neuronowe Seq2Seq

Celem mojego projektu jest napisanie sieci neuronowej, która będzie w stanie znaleźć i poprawić błędy programistyczne w programie użytkownika, napisanej w języku C.

Do wytrenowania sieci neuronowej była wykorzystana baza danych DeepFix (https://paperswithcode.com/dataset/deepfix)

Sieć jest w stanie wykrywać i poprawiać błędy syntaktyczne (np. brak średników, zła ilość nawiasów). Także sieć skutecznie działa w przypadku występowania kilku błędów w jednej linii kodu.

Opis projektu jest zamieszczony w pliku _raport.pdf_.

Przykład działania sieci (więcej przykładów jest w pliku _raport.pdf_):

![](screenshots/oryginalny%20kod.png)

_Rys. 1: oryginalny kod (błędy są zaznaczone na czerwono)_

![](screenshots/poprawiony%20kod.png)

_Rys. 2: kompilacja poprawionego kodu_
