import compiler.CompileBlock;

public class ASTUminus implements ASTNode {

    ASTNode lhs;
    
            public int eval()
            { 
        int v1 = lhs.eval();
            return -v1;
        }
        
            public ASTUminus(ASTNode l)
            {
            lhs = l;
            }

            @Override
        public void compile(CompileBlock c) {
                // TODO Auto-generated method stub
                lhs.compile(c);
                c.emit("sipush -1");
                c.emit("imul");
        }
    }
    
