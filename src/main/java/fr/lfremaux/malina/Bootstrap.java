package fr.lfremaux.malina;

public class Bootstrap {

    private static final String DATA_PATH = "/datasets";

    public static void main(String... args) {
        System.out.println("Hello Max, I'm Malina !");

        new Malina(DATA_PATH);
    }

}
