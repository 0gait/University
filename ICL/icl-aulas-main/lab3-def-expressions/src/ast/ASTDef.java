package ast;

import compiler.FrameCodeBlock;
import compiler.MainCodeBlock;
import util.Coordinates;
import util.Environment;
import util.Pair;

import java.util.*;
public class ASTDef implements ASTNode{
    private List<Pair<String,ASTNode>> init;
    private ASTNode body;
    public ASTDef(List<Pair<String,ASTNode>> init, ASTNode body){
        this.init = init;
        this.body = body;
    }
    @Override
    public int eval(Environment<Integer> e) {
       Iterator<Pair<String,ASTNode>> it = init.iterator();
       Pair<String,ASTNode> pair = null;
       Environment<Integer> env = e.beginScope();
       while(it.hasNext()){
           pair = it.next();
           env.assoc(pair.getLeft(), pair.getRight().eval(env));
       }
       int toReturn = body.eval(env);
       env.endScope();
       return toReturn;
    }

    @Override
    public void compile(MainCodeBlock c, Environment<Coordinates> e) {
        
        Pair<Environment<Coordinates>, Integer> pairEnvFrameId = c.addFrame(this.init.size(), e);
        Environment<Coordinates> newEnv = pairEnvFrameId.getLeft();
        int frameId = pairEnvFrameId.getRight();

        int i = 0;
        for (Pair<String,ASTNode> pair : this.init)
        {
            Coordinates varCoord = new Coordinates(frameId, String.format(FrameCodeBlock.VARIABLE_NAME, i) );

            c.emitCurrentFrame();
            pair.getRight().compile(c, newEnv);
            c.emit(String.format("putfield f%d/%s I", frameId, varCoord.getRight()) );

            newEnv.assoc(pair.getLeft(), varCoord);
            i++;
        }

        this.body.compile(c, newEnv);
        c.endFrame(newEnv);
    }
}
