package ast.bool;

import ast.ASTNodeAbstract;
import compiler.MainCodeBlock;
import types.IType;
import types.primitves.TypeBool;
import util.Coordinates;
import util.Environment;
import values.IValue;
import values.primitive.VBool;

public class ASTBool extends ASTNodeAbstract {

	private boolean val;
	public ASTBool(boolean val) {
		this.val = val;
		type = TypeBool.TYPE;
	}

	public IValue eval(Environment<IValue> e) {
		return new VBool(this.val);
	}
	
	@Override
	public void compile(MainCodeBlock c, Environment<Coordinates> e) {
		if (this.val)
			c.emit("sipush 1");
		else
			c.emit("sipush 0");
	}

	@Override
	public IType typecheck(Environment<IType> e) {
		return type;
	}


}