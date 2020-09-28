all: libget_trend.so
get_trend.o: get_trend.cpp
	g++ get_trend.cpp -c -fPIC
libget_trend.so: get_trend.o
	g++ -shared get_trend.o -o libget_trend.so
clean:
	rm -f get_trend.o libget_trend.so
test:
	python3 get_trend.py
