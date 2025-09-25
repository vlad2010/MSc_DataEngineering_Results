   class MyClass {
   public:
       int myMember;
   };

   int main() {
       // Error: non-static member 'myMember' cannot be used without an object
       int value = MyClass::myMember;
       return 0;
   }