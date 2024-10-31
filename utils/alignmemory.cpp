#include <iostream>

// #pragma pack(8)
alignas(16) struct Member{
    char param;
    int value;
    Member(char param_, int value_) : param(param_), value(value_) {}
};

struct S1
{
    int i:8;    
    char j:4;  
    int a:4;  
    double b;
};

struct S2
{
    int i:8;
    char j:4;  
    double b;
    int a:4;  
};

struct S3
{
    int i;    
    char j;  
    double b;
    int a;     
};


int main()
{
    Member mem('name', 12);


    std::cout << sizeof(S1) << "\n";
    std::cout << sizeof(S2) << "\n";
    std::cout << sizeof(S3) << "\n";
    std::cout << "Sizeof mem: " << sizeof(mem) << "\n" << "char: " << sizeof(char) <<  " - " << "int: " << sizeof(int) << "\n";
    std::cout << "Memory align Done!\n";
}