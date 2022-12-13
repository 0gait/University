package environment;

import java.util.*;
import java.util.Map.Entry;

public class Environment<E>
{
	private Map<String,E> scope;
	private Environment<E> previous;
	
	public Environment (){
		scope = Map.of();
	}

	private Environment(Environment<E> previous) {
		this.previous = previous;
		scope = new HashMap<>();
	}

	public Environment<E> beginScope() {
		return new Environment<>(this);
	}

	public Environment<E> endScope() {
		return previous;
	}

	public void assoc(String id, E value) {
		assert scope != null;

		E aux = scope.putIfAbsent(id, value);
		if (aux != null)
			throw new IllegalArgumentException("Bind already defined for key " + id);
	}

	public E find(String id) {

		E value = scope.get(id);
		if( value != null)
			return value;
		if(previous == null)
			throw new NoSuchElementException(id + " not declared.");
		return previous.find(id);
	}

	public E find(String id, int depth) {

		E value = scope.get(id);
		if (value != null)
			return value;
		if (previous == null || depth == 0)
			throw new NoSuchElementException(id + " not declared.");
		return previous.find(id, depth - 1);
	}

	public int getDepth()
	{
		if (previous == null)
			return 0;
		else
			return previous.getDepth() + 1;
	}

	public Iterable<Entry<String, E>> getLastScope(){
		return scope.entrySet();
	}

	public StringBuilder toString(StringBuilder builder)
	{
		printScopes(builder);
		return builder.append("\n");	
	}

	private int printScopes(StringBuilder builder){
		if(previous == null)
			return 0;

		int id = previous.printScopes(builder) + 1;

		builder.append("\nScope " + id + " = ");
		builder.append(scope.toString());
		return id;
	}

	public String printLastScope()
	{
		return this.scope.toString();
	}
}