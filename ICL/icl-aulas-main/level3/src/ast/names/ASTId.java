package ast.names;

import ast.ASTNodeAbstract;
import compiler.Coordinates;
import compiler.MainCodeBlock;
import environment.Environment;
import typeError.TypeErrorException;
import types.IType;
import values.IValue;

public class ASTId extends ASTNodeAbstract
{
    protected String id;

    public ASTId(String id){
        this.id = id;
    }
    
    @Override
    public IValue eval(Environment<IValue> e) {
        return e.find(id);
    }

    @Override
    public void compile(MainCodeBlock c, Environment<Coordinates> e) {
        
        //Frame atual pode nao ser a depth -1. Frame anterior ao atual pode nao ser o atual -1.

        Coordinates coord = e.find(this.id);
        int varFrameId = coord.getLeft();
        String varId = coord.getRight();
        getFieldFromFrame(c, varFrameId, varId);
    }

    @Override
    public IType typecheck(Environment<IType> e) throws TypeErrorException{
      try{  this.type = e.find(id);
        return type;
        }catch(Exception a){
                throw new TypeErrorException(a.getMessage(), a);
        }
    }

    /**
     * Emits instructions to get a field from the specified frame ID.
     * 
     * In fact, emits the current frame, then reaches the specified frame
     * from the current one and then emits the instruction to get a field.
     * 
     * @param c - The main code block.
     * @param frameId - The id of the frame.
     * @param fieldId - The id of the field.
     */
    protected void getFieldFromFrame(MainCodeBlock c, int frameId, String fieldId)
    {
        c.emitCurrentFrame();
        c.reachFrameIdFromCurrentFrame(frameId);
        c.emit(String.format("getfield f%d/%s %s", frameId, fieldId, getType().getJvmType()) );
    }

    @Override
    public StringBuilder toString(StringBuilder builder) {
        return builder.append(this.id);
    }
}
