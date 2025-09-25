   class MyClass {
   public:
       int myMember;

       static void myStaticFunction() {
           // Error: non-static member 'myMember' cannot be used in a static member function
           int value = myMember;
       }
   };