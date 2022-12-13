package ast.string;

import ast.ASTNode;
import ast.ASTNodeAbstract;
import compiler.Coordinates;
import compiler.MainCodeBlock;
import environment.Environment;
import typeError.TypeErrorException;
import types.IType;
import types.primitives.TypeBool;
import types.primitives.TypeString;
import util.Utils;
import values.IValue;
import values.primitive.VBool;
import values.primitive.VString;

public class ASTIsEmpty extends ASTNodeAbstract{

    private ASTNode string;

    public ASTIsEmpty(ASTNode string){
        this.string = string;
        this.type = TypeBool.TYPE;
    }
    @Override
    public IValue eval(Environment<IValue> e) {
        return 
            new VBool(Utils.checkValueForOperation(string.eval(e), VString.class, "string is empty").getValue().isEmpty());
    }

    @Override
    public void compile(MainCodeBlock c, Environment<Coordinates> e) {
        string.compile(c, e);
        c.emit("invokevirtual java/lang/String/isEmpty()Z");
        
    }

    @Override
    public IType typecheck(Environment<IType> e) throws TypeErrorException {
        Utils.checkTypeForOperation(string.typecheck(e), TypeString.class, "string length");
        return this.type;
    }

    @Override
    public StringBuilder toString(StringBuilder builder) {
        // TODO Auto-generated method stub
        return null;
    }
    
}
