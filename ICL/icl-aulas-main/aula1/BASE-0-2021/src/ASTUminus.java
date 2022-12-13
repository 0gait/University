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
    }
    
