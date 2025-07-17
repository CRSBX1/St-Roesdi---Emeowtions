import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Plus, Edit2, Trash2, Camera } from "lucide-react";
import { useNavigate } from "react-router-dom";
import Navigation from "@/components/shared/Navigation";
import Footer from "@/components/shared/Footer";

interface Pet {
  id: string;
  name: string;
  breed: string;
  age: string;
  photo: string;
  description?: string;
}

const Home = () => {
  const navigate = useNavigate();
  const [pets, setPets] = useState<Pet[]>([
    {
      id: "1",
      name: "Whiskers",
      breed: "Persian",
      age: "3 years",
      photo: "🐱",
      description: "A fluffy and gentle Persian cat who loves to nap in sunny spots."
    },
    {
      id: "2", 
      name: "Shadow",
      breed: "Black Cat",
      age: "2 years",
      photo: "🐈‍⬛",
      description: "A mysterious and playful black cat with bright green eyes."
    }
  ]);

  const [editingPet, setEditingPet] = useState<Pet | null>(null);
  const [isAddDialogOpen, setIsAddDialogOpen] = useState(false);
  const [isEditDialogOpen, setIsEditDialogOpen] = useState(false);
  const [newPet, setNewPet] = useState<Omit<Pet, 'id'>>({
    name: "",
    breed: "",
    age: "",
    photo: "🐱",
    description: ""
  });

  const handleAddPet = () => {
    if (newPet.name && newPet.breed && newPet.age) {
      const pet: Pet = {
        ...newPet,
        id: Date.now().toString()
      };
      setPets([...pets, pet]);
      setNewPet({ name: "", breed: "", age: "", photo: "🐱", description: "" });
      setIsAddDialogOpen(false);
    }
  };

  const handleEditPet = () => {
    if (editingPet) {
      setPets(pets.map(pet => pet.id === editingPet.id ? editingPet : pet));
      setEditingPet(null);
      setIsEditDialogOpen(false);
    }
  };

  const handleDeletePet = (petId: string) => {
    setPets(pets.filter(pet => pet.id !== petId));
  };

  const startEdit = (pet: Pet) => {
    setEditingPet({ ...pet });
    setIsEditDialogOpen(true);
  };

  const catEmojis = ["🐱", "🐈", "🐈‍⬛", "😺", "😸", "😹", "😻", "🙀", "😿", "😾"];

  return (
    <div className="min-h-screen bg-background">
      <Navigation />

      {/* Main Content */}
      <main className="container mx-auto max-w-6xl px-4 py-8">
        <div className="mb-8">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h2 className="text-3xl font-bold text-foreground">Your Cats</h2>
              <p className="text-muted-foreground mt-1">Manage your feline friends and track their emotions</p>
            </div>
            
            <Dialog open={isAddDialogOpen} onOpenChange={setIsAddDialogOpen}>
              <DialogTrigger asChild>
                <Button className="bg-gradient-to-r from-primary to-accent hover:from-primary-hover hover:to-accent-hover shadow-lg">
                  <Plus className="mr-2 h-4 w-4" />
                  Add New Cat
                </Button>
              </DialogTrigger>
              <DialogContent className="sm:max-w-md">
                <DialogHeader>
                  <DialogTitle>Add New Cat</DialogTitle>
                </DialogHeader>
                <div className="space-y-4">
                  <div>
                    <Label htmlFor="name">Name</Label>
                    <Input
                      id="name"
                      value={newPet.name}
                      onChange={(e) => setNewPet({ ...newPet, name: e.target.value })}
                      placeholder="Enter cat's name"
                    />
                  </div>
                  <div>
                    <Label htmlFor="breed">Breed</Label>
                    <Input
                      id="breed"
                      value={newPet.breed}
                      onChange={(e) => setNewPet({ ...newPet, breed: e.target.value })}
                      placeholder="Enter cat's breed"
                    />
                  </div>
                  <div>
                    <Label htmlFor="age">Age</Label>
                    <Input
                      id="age"
                      value={newPet.age}
                      onChange={(e) => setNewPet({ ...newPet, age: e.target.value })}
                      placeholder="e.g., 2 years"
                    />
                  </div>
                  <div>
                    <Label>Photo (Choose an emoji)</Label>
                    <div className="grid grid-cols-5 gap-2 mt-2">
                      {catEmojis.map((emoji) => (
                        <Button
                          key={emoji}
                          variant={newPet.photo === emoji ? "default" : "outline"}
                          className="text-2xl h-12"
                          onClick={() => setNewPet({ ...newPet, photo: emoji })}
                        >
                          {emoji}
                        </Button>
                      ))}
                    </div>
                  </div>
                  <div>
                    <Label htmlFor="description">Description (Optional)</Label>
                    <Textarea
                      id="description"
                      value={newPet.description}
                      onChange={(e) => setNewPet({ ...newPet, description: e.target.value })}
                      placeholder="Tell us about your cat..."
                      rows={3}
                    />
                  </div>
                  <Button onClick={handleAddPet} className="w-full">
                    Add Cat
                  </Button>
                </div>
              </DialogContent>
            </Dialog>
          </div>

          {/* Pet Grid */}
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {pets.map((pet) => (
              <Card key={pet.id} className="hover:shadow-xl transition-all duration-300 cursor-pointer group bg-gradient-to-br from-card to-secondary/20 border-2 hover:border-primary/20">
                <CardHeader className="pb-3">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      <div className="text-4xl">{pet.photo}</div>
                      <div>
                        <CardTitle className="text-lg">{pet.name}</CardTitle>
                        <p className="text-sm text-muted-foreground">{pet.breed}</p>
                      </div>
                    </div>
                    <div className="flex space-x-1 opacity-0 group-hover:opacity-100 transition-opacity">
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-8 w-8"
                        onClick={(e) => {
                          e.stopPropagation();
                          startEdit(pet);
                        }}
                      >
                        <Edit2 className="h-4 w-4" />
                      </Button>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-8 w-8 text-destructive hover:bg-destructive/10"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleDeletePet(pet.id);
                        }}
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                </CardHeader>
                <CardContent
                  onClick={() => navigate(`/analysis/${pet.id}`, { state: { pet } })}
                  className="pt-0"
                >
                  <div className="space-y-2">
                    <Badge variant="secondary">{pet.age}</Badge>
                    {pet.description && (
                      <p className="text-sm text-muted-foreground line-clamp-2">
                        {pet.description}
                      </p>
                    )}
                    <Button variant="outline" className="w-full mt-4 bg-gradient-to-r from-accent/20 to-success/20 hover:from-accent/30 hover:to-success/30 border-accent">
                      <Camera className="mr-2 h-4 w-4" />
                      Analyze Emotions
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>

          {pets.length === 0 && (
            <div className="text-center py-12">
              <div className="text-6xl mb-4">🐱</div>
              <h3 className="text-xl font-semibold text-foreground mb-2">No cats added yet</h3>
              <p className="text-muted-foreground mb-4">Add your first cat to start analyzing their emotions!</p>
              <Button onClick={() => setIsAddDialogOpen(true)} className="bg-gradient-to-r from-primary to-accent hover:from-primary-hover hover:to-accent-hover shadow-lg">
                <Plus className="mr-2 h-4 w-4" />
                Add Your First Cat
              </Button>
            </div>
          )}
        </div>
      </main>

      <Footer />

      {/* Edit Dialog */}
      <Dialog open={isEditDialogOpen} onOpenChange={setIsEditDialogOpen}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>Edit Cat Details</DialogTitle>
          </DialogHeader>
          {editingPet && (
            <div className="space-y-4">
              <div>
                <Label htmlFor="edit-name">Name</Label>
                <Input
                  id="edit-name"
                  value={editingPet.name}
                  onChange={(e) => setEditingPet({ ...editingPet, name: e.target.value })}
                />
              </div>
              <div>
                <Label htmlFor="edit-breed">Breed</Label>
                <Input
                  id="edit-breed"
                  value={editingPet.breed}
                  onChange={(e) => setEditingPet({ ...editingPet, breed: e.target.value })}
                />
              </div>
              <div>
                <Label htmlFor="edit-age">Age</Label>
                <Input
                  id="edit-age"
                  value={editingPet.age}
                  onChange={(e) => setEditingPet({ ...editingPet, age: e.target.value })}
                />
              </div>
              <div>
                <Label>Photo</Label>
                <div className="grid grid-cols-5 gap-2 mt-2">
                  {catEmojis.map((emoji) => (
                    <Button
                      key={emoji}
                      variant={editingPet.photo === emoji ? "default" : "outline"}
                      className="text-2xl h-12"
                      onClick={() => setEditingPet({ ...editingPet, photo: emoji })}
                    >
                      {emoji}
                    </Button>
                  ))}
                </div>
              </div>
              <div>
                <Label htmlFor="edit-description">Description</Label>
                <Textarea
                  id="edit-description"
                  value={editingPet.description || ""}
                  onChange={(e) => setEditingPet({ ...editingPet, description: e.target.value })}
                  rows={3}
                />
              </div>
              <Button onClick={handleEditPet} className="w-full">
                Save Changes
              </Button>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default Home;