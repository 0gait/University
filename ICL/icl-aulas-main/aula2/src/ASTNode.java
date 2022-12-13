import compiler.CompileBlock;

public interface ASTNode {

    int eval();

    void compile(CompileBlock c);
	
}

