package ast.arithmetic;

import ast.ASTNode;
import compiler.MainCodeBlock;
import util.Coordinates;
import util.Environment;

public class ASTUminus implements ASTNode {

    ASTNode lhs;

    public int eval(Environment<Integer> e) {
        int v1 = lhs.eval(e);
        return -v1;
    }

    public ASTUminus(ASTNode l) {
        lhs = l;
    }

    @Override
    public void compile(MainCodeBlock c, Environment<Coordinates> e) {
        lhs.compile(c, e);
        c.emit("sipush -1");
        c.emit("imul");
    }
}
