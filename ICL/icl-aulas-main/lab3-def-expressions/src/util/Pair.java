package util;

public class Pair<K,V> {
    private K item1;
    private V item2;

    public Pair(K item1, V item2){
        this.item1 = item1;
        this.item2 = item2;
    }

    public K getLeft(){
        return item1;
    }

    public V getRight(){
        return item2;
    }
}
