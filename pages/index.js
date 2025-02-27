import { useRouter } from 'next/router';

export default function Home() {
  const router = useRouter();
  const townPeople = ["Alice", "Bob", "Charlie", "Diana", "Ethan", "Fiona"];

  const selectPerson = (person) => {
    router.push(`/chat?person=${person}`);
  };

  return (
    <div className="flex flex-col items-center justify-center h-screen">
      <h1 className="text-2xl font-bold">Select a Town Person</h1>
      <div className="mt-4">
        {townPeople.map((person) => (
          <button 
            key={person} 
            onClick={() => selectPerson(person)}
            className="m-2 px-4 py-2 bg-blue-500 text-white rounded-lg"
          >
            {person}
          </button>
        ))}
      </div>
    </div>
  );
}
