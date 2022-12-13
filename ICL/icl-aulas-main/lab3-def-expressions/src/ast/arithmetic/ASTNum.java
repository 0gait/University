package ast.arithmetic;

import ast.ASTNode;
import compiler.MainCodeBlock;
import util.Coordinates;
import util.Environment;

public class ASTNum implements ASTNode {

        int val;

        public int eval(Environment<Integer> e) {
                return val;
        }

        public ASTNum(int n) {
                val = n;
        }

        @Override
        public void compile(MainCodeBlock c, Environment<Coordinates> e) {
                c.emit("sipush " + val);
        }

}
