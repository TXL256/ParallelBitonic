#include <iostream>
#include <fstream>


int * read_file(char * name, int * len) {
    std::ifstream infile(name);

    infile >> *len;

    int * data = (int *)malloc(sizeof(int) * *len);

    for (int i = 0; i < *len; i++) {
        infile >> data[i];
    }

    infile.close();

    return data;
}
